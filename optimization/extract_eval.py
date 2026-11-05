"""Parse scalar native evaluation logs without evaluating executable expressions."""
import argparse,ast,json,math,re
from pathlib import Path

def value(node):
 if isinstance(node,ast.Constant) and type(node.value) in (str,int,float):return node.value
 if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
  v=value(node.operand)
  if type(v) not in (int,float):raise ValueError('Nonnumeric unary operand')
  return -v if isinstance(node.op,ast.USub) else v
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='np' and node.func.attr in ('float64','float32','int64','int32') and len(node.args)==1 and not node.keywords:
  v=value(node.args[0])
  if type(v) not in (int,float):raise ValueError('Nonnumeric scalar constructor')
  return v
 raise ValueError('Unsupported log expression')

def extract(text,step):
 metrics={};records=0
 for line in text.splitlines():
  match=re.search(r'Step (\d+), eval ([^:]+): (\{.*\})',line)
  if not match or int(match[1])!=step:continue
  tree=ast.parse(match[3],mode='eval').body
  if not isinstance(tree,ast.Dict):raise ValueError('Expected dictionary')
  for k,v in zip(tree.keys,tree.values):
   key=match[2]+'::'+value(k);val=value(v)
   if type(val) not in (int,float) or not math.isfinite(val):raise ValueError('Nonfinite or nonscalar eval value')
   if key in metrics:raise ValueError('Duplicate eval metric')
   metrics[key]=val
  records+=1
 if records<2 or not any('/ema0.999_' in k for k in metrics) or not any('/ema0.999_' not in k for k in metrics):raise ValueError('Missing normal or EMA evaluation')
 return {'step':step,'records':records,'metrics':metrics,'scope':'Native evaluation on configured fixture, not full-corpus accuracy'}

def compare(reference,candidate):
 if reference['step']!=candidate['step'] or set(reference['metrics'])!=set(candidate['metrics']):raise ValueError('Evaluation scope or metric coverage differs')
 rows=[]
 for k,r in reference['metrics'].items():
  c=candidate['metrics'][k];limit=max(.0001,.05*abs(r))
  rows.append({'metric':k,'reference':r,'candidate':c,'delta':abs(c-r),'limit':limit,'passed':abs(c-r)<=limit})
 return {'evaluation_screen_passed':all(r['passed'] for r in rows),'metrics':rows}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('--step',required=True,type=int);p.add_argument('--output',required=True);a=p.parse_args()
 report=extract(Path(a.log).read_text(),a.step)
 with open(a.output,'x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'records':report['records'],'metrics':len(report['metrics'])}))
