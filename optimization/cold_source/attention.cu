#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#include <type_traits>
using namespace PyTorchMemEffAttention;
template<int M,int N> std::vector<at::Tensor> launch(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias){
 using AK=AttentionKernel<float,cutlass::arch::Sm50,false,M,N,64,false,true>;
 static_assert(std::is_same<typename AK::MM0::GemmType::OpClass,cutlass::arch::OpClassSimt>::value);
 static_assert(std::is_same<typename AK::MM1::GemmType::OpClass,cutlass::arch::OpClassSimt>::value);
 static_assert(std::is_same<typename AK::MM0::GemmType::Operator,cutlass::arch::OpMultiplyAdd>::value);
 static_assert(std::is_same<typename AK::MM1::GemmType::Operator,cutlass::arch::OpMultiplyAdd>::value);
 // QKV are [B,H,T,D]. Upstream kernel requires output storage [B,T,H,D].
 auto output=at::empty({q.size(0),q.size(2),q.size(1),v.size(3)},q.options());
 // BiasLoader always uses vector4 loads, even for unaligned QKV templates.
 // Allocate padded row storage so N379 has aligned rows and valid tail reads.
 auto storage=at::zeros({1,q.size(1),q.size(2),((q.size(2)+3)/4)*4},bias.options());
 auto b=storage.narrow(3,0,q.size(2));b.copy_(bias);
 TORCH_CHECK(b.stride(2)%4==0 && b.stride(1)%4==0); // Copy + padding counted.
 auto lse=at::empty({q.size(0),q.size(1),((q.size(2)+31)/32)*32},q.options());
 typename AK::Params p;p.logsumexp_ptr=lse.data_ptr<float>();
 p.query_ptr=q.data_ptr<float>();p.key_ptr=k.data_ptr<float>();p.value_ptr=v.data_ptr<float>();p.attn_bias_ptr=b.data_ptr<float>();p.output_ptr=output.data_ptr<float>();
 p.scale=1.0f;p.head_dim=q.size(3);p.head_dim_value=v.size(3);p.num_queries=q.size(2);p.num_keys=k.size(2);p.num_batches=q.size(0);p.num_heads=q.size(1);
 p.q_strideM=q.stride(2);p.k_strideM=k.stride(2);p.v_strideM=v.stride(2);p.o_strideM=output.stride(1);
 p.q_strideH=q.stride(1);p.k_strideH=k.stride(1);p.v_strideH=v.stride(1);
 p.q_strideB=q.stride(0);p.k_strideB=k.stride(0);p.v_strideB=v.stride(0);
 p.bias_strideM=b.stride(2);p.bias_strideH=b.stride(1);p.bias_strideB=0;
 AK::check_supported(p);
 auto kernel=attention_kernel_batched_impl<AK>;size_t smem=sizeof(typename AK::SharedStorage);
 if(smem>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 kernel<<<p.getBlocksGrid(),p.getThreadsGrid(),smem,c10::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {output.transpose(1,2),lse,b};
}
std::vector<at::Tensor> simt_forward(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor b,int mode){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kFloat&&q.dim()==4&&q.size(3)==48);
 for(auto t:{k,v})TORCH_CHECK(t.device()==q.device()&&t.scalar_type()==at::kFloat&&t.sizes()==q.sizes()&&t.stride(3)==1);
 TORCH_CHECK(q.stride(3)==1&&b.device()==q.device()&&b.scalar_type()==at::kFloat&&b.dim()==4&&b.size(0)==1&&b.size(1)==q.size(1)&&b.size(2)==q.size(2)&&b.size(3)==q.size(2));

 c10::cuda::CUDAGuard guard(q.device());
 if(mode==0)return launch<32,64>(q,k,v,b);
 if(mode==1)return launch<64,64>(q,k,v,b);
 TORCH_CHECK(false,"unsupported mode");
}
std::vector<at::Tensor> simt_backward(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor b,at::Tensor output,at::Tensor lse,at::Tensor grad){
 using AK=AttentionBackwardKernel<cutlass::arch::Sm50,float,false,false,false,64,64,64,false>;
 static_assert(std::is_same<typename AK::GemmType::OpClass,cutlass::arch::OpClassSimt>::value);
 static_assert(std::is_same<typename AK::GemmType::Operator,cutlass::arch::OpMultiplyAdd>::value);
 static_assert(!AK::kKernelComputesDelta);
 c10::cuda::CUDAGuard guard(q.device());
 auto B=q.size(0),H=q.size(1),T=q.size(2),D=q.size(3);
 for(auto x:{q,k,v,b,output,lse,grad})TORCH_CHECK(x.is_cuda()&&x.device()==q.device()&&x.scalar_type()==at::kFloat);
 TORCH_CHECK(q.dim()==4&&D==48&&k.sizes()==q.sizes()&&v.sizes()==q.sizes()&&output.sizes()==q.sizes()&&grad.sizes()==q.sizes());
 TORCH_CHECK(output.stride(1)==D&&output.stride(2)==H*D&&b.stride(3)==1&&b.stride(2)%4==0);
 auto go=grad.transpose(1,2).contiguous().transpose(1,2);
 auto delta=(output*go).sum(-1).contiguous();
 auto dq=at::empty({B,T,H,D},q.options()),dk=at::empty({B,T,H,D},q.options()),dv=at::empty({B,T,H,D},q.options());
 // Per-example bias gradients must be reduced across the broadcast batch.
 auto dbstorage=at::zeros({B,H,T,((T+3)/4)*4},q.options());auto db=dbstorage.narrow(3,0,T);
 typename AK::Params p;
 p.query_ptr=q.data_ptr<float>();p.key_ptr=k.data_ptr<float>();p.value_ptr=v.data_ptr<float>();p.bias_ptr=b.data_ptr<float>();p.output_ptr=output.data_ptr<float>();p.logsumexp_ptr=lse.data_ptr<float>();p.grad_output_ptr=go.data_ptr<float>();p.delta_ptr=delta.data_ptr<float>();
 p.grad_query_ptr=dq.data_ptr<float>();p.grad_key_ptr=dk.data_ptr<float>();p.grad_value_ptr=dv.data_ptr<float>();p.grad_bias_ptr=db.data_ptr<float>();
 p.head_dim=D;p.head_dim_value=D;p.num_queries=T;p.num_keys=T;p.num_heads=H;p.num_batches=B;p.scale=1.0f;p.num_splits_key=1;
 p.q_strideM=q.stride(2);p.k_strideM=k.stride(2);p.v_strideM=v.stride(2);p.gO_strideM=go.stride(2);p.bias_strideM=b.stride(2);p.gB_strideM=db.stride(2);
 p.q_strideH=q.stride(1);p.k_strideH=k.stride(1);p.v_strideH=v.stride(1);p.q_strideB=q.stride(0);p.k_strideB=k.stride(0);p.v_strideB=v.stride(0);
 p.o_strideH=output.stride(1);p.o_strideB=output.stride(0);p.gO_strideH=go.stride(1);p.gO_strideB=go.stride(0);
 p.bias_strideH=b.stride(1);p.bias_strideB=0;p.lse_strideH=lse.stride(1);p.lse_strideB=lse.stride(0);p.delta_strideH=delta.stride(1);p.delta_strideB=delta.stride(0);
 p.gQ_strideH=D;p.gK_strideH=D;p.gV_strideH=D;p.gQ_strideB=dq.stride(0);p.gK_strideB=dk.stride(0);p.gV_strideB=dv.stride(0);p.gB_strideH=db.stride(1);p.gB_strideB=db.stride(0);
 auto workspace=at::zeros({p.workspace_size()/4},q.options());p.workspace=workspace.numel()?workspace.data_ptr<float>():nullptr;
 AK::check_supported(p);
 auto kernel=attention_kernel_backward_batched_impl<AK>;size_t smem=sizeof(typename AK::SharedStorage);
 if(smem>48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 kernel<<<p.getBlocksGrid(),p.getThreadsGrid(),smem,c10::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {dq.transpose(1,2),dk.transpose(1,2),dv.transpose(1,2),db.sum(0,true)};
}
