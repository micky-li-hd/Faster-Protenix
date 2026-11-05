#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace width64 {
// Two-stage affine reduction preserving native per-lane and final sum order.
template<typename T,int RowLanes>
__global__ void partial_affine(const T* x,const T* dy,const float* mean,
 const float* rstd,float* partial,int64_t rows,int64_t chunks,bool wg,bool wb) {
 const int col=blockIdx.x*32+threadIdx.x;
 const int lane=blockIdx.y*RowLanes+threadIdx.y;
 float gs=0.0f,bs=0.0f;
 // As in native GammaBetaBackwardCUDAKernel_32x32, the bulk loop
 // has no per-row tail predicate. Each native lane keeps its exact order.
 const int64_t bulk=rows/256;
 const int64_t grouped_bulk=(bulk/4)*4;
 for(int64_t first=0;first<grouped_bulk;first+=4) {
  float loaded_mean[4]={0.0f,0.0f,0.0f,0.0f};
  float loaded_rstd[4]={0.0f,0.0f,0.0f,0.0f};
  float loaded_x[4][8],loaded_dy[4][8];
  #pragma unroll
  for(int group=0;group<4;++group) {
   int64_t offset=((first+group)*32+lane)*8;
   if(threadIdx.x<8) {
    loaded_mean[group]=mean[offset+threadIdx.x];
    loaded_rstd[group]=rstd[offset+threadIdx.x];
   }
   #pragma unroll
   for(int ii=0;ii<8;++ii) {
    int64_t row=offset+ii;
    loaded_dy[group][ii]=float(dy[row*64+col]);
    loaded_x[group][ii]=wg?float(x[row*64+col]):0.0f;
   }
  }
  __syncwarp(0xffffffff);
  // Consume in the original per-lane chunk,row order, with no partial sums.
  #pragma unroll
  for(int group=0;group<4;++group) {
   #pragma unroll
   for(int ii=0;ii<8;++ii) {
    float m=__shfl_sync(0xffffffff,loaded_mean[group],ii,32);
    float rs=__shfl_sync(0xffffffff,loaded_rstd[group],ii,32);
    float gy=loaded_dy[group][ii];
    if(wg)gs+=gy*(loaded_x[group][ii]-m)*rs;
    if(wb)bs+=gy;
   }
  }
 }
 for(int64_t chunk=grouped_bulk;chunk<bulk;++chunk) {
  int64_t offset=(chunk*32+lane)*8;
  float lm=0.0f,lr=0.0f;
  if(threadIdx.x<8) {
   lm=mean[offset+threadIdx.x];lr=rstd[offset+threadIdx.x];
  }
  __syncwarp(0xffffffff);
  #pragma unroll
  for(int ii=0;ii<8;++ii) {
   float m=__shfl_sync(0xffffffff,lm,ii,32);
   float rs=__shfl_sync(0xffffffff,lr,ii,32);
   int64_t row=offset+ii;
   float gy=float(dy[row*64+col]);
   if(wg)gs+=gy*(float(x[row*64+col])-m)*rs;
   if(wb)bs+=gy;
  }
 }
 int64_t offset=(bulk*32+lane)*8;
 for(int ii=0;ii<8;++ii) {
  int64_t row=offset+ii;
  if(row<rows) {
   float gy=float(dy[row*64+col]);
   if(wg)gs+=gy*(float(x[row*64+col])-mean[row])*rstd[row];
   if(wb)bs+=gy;
  }
 }
 if(wg)partial[lane*64+col]=gs;
 if(wb)partial[32*64+lane*64+col]=bs;
}
template<typename T>
__global__ void finish_affine(const float* partial,T* dg,T* db) {
 __shared__ float sg[32][32];
 __shared__ float sb[32][32];
 int col=blockIdx.x*32+threadIdx.x,lane=threadIdx.y;
 sg[lane][threadIdx.x]=dg?partial[lane*64+col]:0.0f;
 sb[lane][threadIdx.x]=db?partial[32*64+lane*64+col]:0.0f;
 __syncthreads();
 for(int delta=16;delta>=1;delta>>=1) {
  if(lane<delta) {
   sg[lane][threadIdx.x]+=sg[lane+delta][threadIdx.x];
   sb[lane][threadIdx.x]+=sb[lane+delta][threadIdx.x];
  }
  __syncthreads();
 }
 if(lane==0) {
  if(dg)dg[col]=T(sg[0][threadIdx.x]);
  if(db)db[col]=T(sb[0][threadIdx.x]);
 }
}
template<typename T,int RowLanes>
void launch_group(const torch::Tensor& x,const torch::Tensor& dy,
 const torch::Tensor& mean,const torch::Tensor& rstd,torch::Tensor& scratch,
 torch::Tensor& dg,torch::Tensor& db,int64_t rows,int64_t chunks) {
 auto stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
 partial_affine<T,RowLanes><<<dim3(2,32/RowLanes),dim3(32,RowLanes),0,stream>>>(x.data_ptr<T>(),dy.data_ptr<T>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),scratch.data_ptr<float>(),rows,chunks,dg.defined(),db.defined());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 finish_affine<T><<<2,dim3(32,32),0,stream>>>(scratch.data_ptr<float>(),dg.defined()?dg.data_ptr<T>():nullptr,db.defined()?db.data_ptr<T>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename T>
void launch(const torch::Tensor& x,const torch::Tensor& dy,const torch::Tensor& mean,const torch::Tensor& rstd,torch::Tensor& scratch,torch::Tensor& dg,torch::Tensor& db,int64_t rows,int64_t chunks,int64_t lanes) {
 if(lanes==1)launch_group<T,1>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else if(lanes==2)launch_group<T,2>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else if(lanes==4)launch_group<T,4>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else launch_group<T,8>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
}
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor x,torch::Tensor dy,torch::Tensor mean,torch::Tensor rstd,bool want_weight,bool want_bias,int64_t lanes) {
 TORCH_CHECK(lanes==1 || lanes==2 || lanes==4 || lanes==8,"row lanes must be1,2,4or8");
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim()>=2 && x.size(-1)==64,"x must be contiguous CUDA (...,64)");
 TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kBFloat16,"FP32/BF16 only");
 TORCH_CHECK(dy.device()==x.device() && dy.scalar_type()==x.scalar_type() && dy.sizes()==x.sizes() && dy.is_contiguous(),"dy mismatch");
 int64_t rows=x.numel()/64,chunks=(rows+255)/256;
 TORCH_CHECK(rows>=131072 && rows<=4194304 && rows%32==0,"unsupported row count");
 for(const auto& t:{mean,rstd})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kFloat && t.is_contiguous() && t.numel()==rows,"stats mismatch");
 c10::cuda::CUDAGuard guard(x.device());torch::Tensor dg,db;
 if(want_weight)dg=torch::empty({64},x.options());
 if(want_bias)db=torch::empty({64},x.options());
 if(want_weight || want_bias) {
  auto scratch=torch::empty({2,32,64},x.options().dtype(at::kFloat));
  if(x.scalar_type()==at::kFloat)launch<float>(x,dy,mean,rstd,scratch,dg,db,rows,chunks,lanes);
  else launch<c10::BFloat16>(x,dy,mean,rstd,scratch,dg,db,rows,chunks,lanes);
 }
 return {dg,db};
}
}
namespace width128 {
// Two-stage affine reduction preserving native per-lane and final sum order.
template<typename T,int RowLanes>
__global__ void partial_affine(const T* x,const T* dy,const float* mean,
 const float* rstd,float* partial,int64_t rows,int64_t chunks,bool wg,bool wb) {
 const int col=blockIdx.x*32+threadIdx.x;
 const int lane=blockIdx.y*RowLanes+threadIdx.y;
 float gs=0.0f,bs=0.0f;
 // As in native GammaBetaBackwardCUDAKernel_32x32, the bulk loop
 // has no per-row tail predicate. Each native lane keeps its exact order.
 const int64_t bulk=rows/256;
 const int64_t grouped_bulk=(bulk/4)*4;
 for(int64_t first=0;first<grouped_bulk;first+=4) {
  float loaded_mean[4]={0.0f,0.0f,0.0f,0.0f};
  float loaded_rstd[4]={0.0f,0.0f,0.0f,0.0f};
  float loaded_x[4][8],loaded_dy[4][8];
  #pragma unroll
  for(int group=0;group<4;++group) {
   int64_t offset=((first+group)*32+lane)*8;
   if(threadIdx.x<8) {
    loaded_mean[group]=mean[offset+threadIdx.x];
    loaded_rstd[group]=rstd[offset+threadIdx.x];
   }
   #pragma unroll
   for(int ii=0;ii<8;++ii) {
    int64_t row=offset+ii;
    loaded_dy[group][ii]=float(dy[row*128+col]);
    loaded_x[group][ii]=wg?float(x[row*128+col]):0.0f;
   }
  }
  __syncwarp(0xffffffff);
  // Consume in the original per-lane chunk,row order, with no partial sums.
  #pragma unroll
  for(int group=0;group<4;++group) {
   #pragma unroll
   for(int ii=0;ii<8;++ii) {
    float m=__shfl_sync(0xffffffff,loaded_mean[group],ii,32);
    float rs=__shfl_sync(0xffffffff,loaded_rstd[group],ii,32);
    float gy=loaded_dy[group][ii];
    if(wg)gs+=gy*(loaded_x[group][ii]-m)*rs;
    if(wb)bs+=gy;
   }
  }
 }
 for(int64_t chunk=grouped_bulk;chunk<bulk;++chunk) {
  int64_t offset=(chunk*32+lane)*8;
  float lm=0.0f,lr=0.0f;
  if(threadIdx.x<8) {
   lm=mean[offset+threadIdx.x];lr=rstd[offset+threadIdx.x];
  }
  __syncwarp(0xffffffff);
  #pragma unroll
  for(int ii=0;ii<8;++ii) {
   float m=__shfl_sync(0xffffffff,lm,ii,32);
   float rs=__shfl_sync(0xffffffff,lr,ii,32);
   int64_t row=offset+ii;
   float gy=float(dy[row*128+col]);
   if(wg)gs+=gy*(float(x[row*128+col])-m)*rs;
   if(wb)bs+=gy;
  }
 }
 int64_t offset=(bulk*32+lane)*8;
 for(int ii=0;ii<8;++ii) {
  int64_t row=offset+ii;
  if(row<rows) {
   float gy=float(dy[row*128+col]);
   if(wg)gs+=gy*(float(x[row*128+col])-mean[row])*rstd[row];
   if(wb)bs+=gy;
  }
 }
 if(wg)partial[lane*128+col]=gs;
 if(wb)partial[32*128+lane*128+col]=bs;
}
template<typename T>
__global__ void finish_affine(const float* partial,T* dg,T* db) {
 __shared__ float sg[32][32];
 __shared__ float sb[32][32];
 int col=blockIdx.x*32+threadIdx.x,lane=threadIdx.y;
 sg[lane][threadIdx.x]=dg?partial[lane*128+col]:0.0f;
 sb[lane][threadIdx.x]=db?partial[32*128+lane*128+col]:0.0f;
 __syncthreads();
 for(int delta=16;delta>=1;delta>>=1) {
  if(lane<delta) {
   sg[lane][threadIdx.x]+=sg[lane+delta][threadIdx.x];
   sb[lane][threadIdx.x]+=sb[lane+delta][threadIdx.x];
  }
  __syncthreads();
 }
 if(lane==0) {
  if(dg)dg[col]=T(sg[0][threadIdx.x]);
  if(db)db[col]=T(sb[0][threadIdx.x]);
 }
}
template<typename T,int RowLanes>
void launch_group(const torch::Tensor& x,const torch::Tensor& dy,
 const torch::Tensor& mean,const torch::Tensor& rstd,torch::Tensor& scratch,
 torch::Tensor& dg,torch::Tensor& db,int64_t rows,int64_t chunks) {
 auto stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
 partial_affine<T,RowLanes><<<dim3(4,32/RowLanes),dim3(32,RowLanes),0,stream>>>(x.data_ptr<T>(),dy.data_ptr<T>(),mean.data_ptr<float>(),rstd.data_ptr<float>(),scratch.data_ptr<float>(),rows,chunks,dg.defined(),db.defined());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 finish_affine<T><<<4,dim3(32,32),0,stream>>>(scratch.data_ptr<float>(),dg.defined()?dg.data_ptr<T>():nullptr,db.defined()?db.data_ptr<T>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename T>
void launch(const torch::Tensor& x,const torch::Tensor& dy,const torch::Tensor& mean,const torch::Tensor& rstd,torch::Tensor& scratch,torch::Tensor& dg,torch::Tensor& db,int64_t rows,int64_t chunks,int64_t lanes) {
 if(lanes==1)launch_group<T,1>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else if(lanes==2)launch_group<T,2>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else if(lanes==4)launch_group<T,4>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
 else launch_group<T,8>(x,dy,mean,rstd,scratch,dg,db,rows,chunks);
}
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor x,torch::Tensor dy,torch::Tensor mean,torch::Tensor rstd,bool want_weight,bool want_bias,int64_t lanes) {
 TORCH_CHECK(lanes==1 || lanes==2 || lanes==4 || lanes==8,"row lanes must be1,2,4or8");
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim()>=2 && x.size(-1)==128,"x must be contiguous CUDA (...,128)");
 TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kBFloat16,"FP32/BF16 only");
 TORCH_CHECK(dy.device()==x.device() && dy.scalar_type()==x.scalar_type() && dy.sizes()==x.sizes() && dy.is_contiguous(),"dy mismatch");
 int64_t rows=x.numel()/128,chunks=(rows+255)/256;
 TORCH_CHECK(rows>=131072 && rows<=4194304 && rows%32==0,"unsupported row count");
 for(const auto& t:{mean,rstd})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kFloat && t.is_contiguous() && t.numel()==rows,"stats mismatch");
 c10::cuda::CUDAGuard guard(x.device());torch::Tensor dg,db;
 if(want_weight)dg=torch::empty({128},x.options());
 if(want_bias)db=torch::empty({128},x.options());
 if(want_weight || want_bias) {
  auto scratch=torch::empty({2,32,128},x.options().dtype(at::kFloat));
  if(x.scalar_type()==at::kFloat)launch<float>(x,dy,mean,rstd,scratch,dg,db,rows,chunks,lanes);
  else launch<c10::BFloat16>(x,dy,mean,rstd,scratch,dg,db,rows,chunks,lanes);
 }
 return {dg,db};
}
}
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor x,torch::Tensor dy,torch::Tensor mean,torch::Tensor rstd,bool want_weight,bool want_bias,int64_t lanes) {
 TORCH_CHECK(x.dim()>=2,"x must have at least2 dimensions");
 TORCH_CHECK(x.size(-1)==64 || x.size(-1)==128,"unsupported width");
 if(x.size(-1)==64)return width64::affine_backward_cuda(x,dy,mean,rstd,want_weight,want_bias,lanes);
 return width128::affine_backward_cuda(x,dy,mean,rstd,want_weight,want_bias,lanes);
}

namespace width16 {
// Each native y-lane retains its original consecutive eight-row groups.
template<int R>
__global__ void partial(const float* x,const float* dy,const float* mean,const float* rs,float* scratch,int64_t rows){
 int col=threadIdx.x,lane=blockIdx.x*R+threadIdx.y;
 float gs=0,bs=0;int chunk;
 for(chunk=0;chunk<rows/(32*8);++chunk){
  int offset=(chunk*32+lane)*8;
  #pragma unroll
  for(int ii=0;ii<8;++ii){
   float gy=dy[(offset+ii)*16+col];
   gs+=gy*(x[(offset+ii)*16+col]-mean[offset+ii])*rs[offset+ii];
   bs+=gy;
  }
 }
 int offset=(chunk*32+lane)*8;
 for(int ii=0;ii<8;++ii)if(offset+ii<rows){
  float gy=dy[(offset+ii)*16+col];
  gs+=gy*(x[(offset+ii)*16+col]-mean[offset+ii])*rs[offset+ii];bs+=gy;
 }
 scratch[lane*16+col]=gs;scratch[512+lane*16+col]=bs;
}
__global__ void finish(const float* scratch,float* dg,float* db){
 __shared__ float g[32][16],b[32][16];int col=threadIdx.x,lane=threadIdx.y;
 g[lane][col]=scratch[lane*16+col];b[lane][col]=scratch[512+lane*16+col];__syncthreads();
 for(int offset=16;offset>=1;offset/=2){
  if(lane<offset){g[lane][col]+=g[lane+offset][col];b[lane][col]+=b[lane+offset][col];}
  __syncthreads();
 }
 if(lane==0){if(dg)dg[col]=g[0][col];if(db)db[col]=b[0][col];}
}
template<int R>void launch(torch::Tensor x,torch::Tensor dy,torch::Tensor mean,torch::Tensor rs,torch::Tensor scratch,torch::Tensor dg,torch::Tensor db){
 auto stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
 partial<R><<<32/R,dim3(16,R),0,stream>>>(x.data_ptr<float>(),dy.data_ptr<float>(),mean.data_ptr<float>(),rs.data_ptr<float>(),scratch.data_ptr<float>(),x.numel()/16);C10_CUDA_KERNEL_LAUNCH_CHECK();
 finish<<<1,dim3(16,32),0,stream>>>(scratch.data_ptr<float>(),dg.defined()?dg.data_ptr<float>():nullptr,db.defined()?db.data_ptr<float>():nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<torch::Tensor> affine_backward_cuda(torch::Tensor x,torch::Tensor dy,torch::Tensor mean,torch::Tensor rs,bool wg,bool wb,int64_t lanes){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==at::kFloat&&x.dim()>=2&&x.size(-1)==16,"FP32 contiguous width16 required");
 TORCH_CHECK(x.numel()/16>=128&&x.numel()/16<=4194304,"row bounds");
 TORCH_CHECK(dy.device()==x.device()&&dy.scalar_type()==x.scalar_type()&&dy.sizes()==x.sizes()&&dy.is_contiguous(),"dy mismatch");
 for(const auto&t:{mean,rs})TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous()&&t.numel()==x.numel()/16,"stats mismatch");
 TORCH_CHECK(lanes==1||lanes==2||lanes==4||lanes==8,"lanes");
 c10::cuda::CUDAGuard guard(x.device());torch::Tensor dg,db;
 if(wg)dg=torch::empty({16},x.options());if(wb)db=torch::empty({16},x.options());
 if(wg||wb){auto s=torch::empty({2,32,16},x.options());
 if(lanes==1)launch<1>(x,dy,mean,rs,s,dg,db);else if(lanes==2)launch<2>(x,dy,mean,rs,s,dg,db);else if(lanes==4)launch<4>(x,dy,mean,rs,s,dg,db);else launch<8>(x,dy,mean,rs,s,dg,db);}
 return {dg,db};
}
}
