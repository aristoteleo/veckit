"""Compare mixed-task threaded scoring with serial scoring on invented temporary data."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,platform,sys,tempfile,threading,types,warnings
import anndata as ad
import numpy as np
import pandas as pd

ALIASES=('metrics','metrics_v2','_reuse','t1_metrics','t2_metrics')


def make_files(folder):
    rng=np.random.default_rng(42);n,g=96,40
    labels=np.repeat(['invented_a','invented_b','invented_c'],32)
    x=rng.gamma(2,1,(n,g)).astype(np.float32)
    x[labels=='invented_b',:10]+=1
    x[labels=='invented_c',10:20]+=1
    coords=rng.normal(size=(n,3)).astype(np.float32)*[1,2,3]
    paths={}
    for role,scale in [('reference',1.),('target',1.1),('input',1.05)]:
        expression=x.copy()
        if role!='reference':
            expression[:,:12]*=2.0 if role=='target' else 1.7
            expression[:,12:24]*=.4 if role=='target' else .6
        a=ad.AnnData(expression,obs=pd.DataFrame({'celltype':labels},index=[f'cell{i}' for i in range(n)]),var=pd.DataFrame(index=[f'invented_gene{i}' for i in range(g)]))
        a.obsm['spatial_3D']=(coords*scale).astype(np.float32)
        path=folder/f'{role}.h5ad';a.write_h5ad(path);paths[role]=path
    return paths


def numeric_equal(left,right):
    if left.keys()!=right.keys():return False
    for key,value in left.items():
        other=right[key]
        if value is None or other is None:
            if value is not other:return False
        elif not np.isclose(value,other,rtol=0,atol=0,equal_nan=True):return False
    return True


def run(source,calls=12,workers=6,rounds=3):
    sys.path.insert(0,str(source.resolve()))
    import veckit
    assert Path(veckit.__file__).resolve()==source.resolve()/'veckit.py'
    # Existing applications may already own these generic module names.
    sentinels={name:types.ModuleType(name) for name in ALIASES}
    for name,module in sentinels.items():sys.modules[name]=module
    path_before=list(sys.path)
    with tempfile.TemporaryDirectory(prefix='task-isolation-') as td,warnings.catch_warnings():
        warnings.simplefilter('ignore');paths=make_files(Path(td))
        before={role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in paths.items()}
        def score(task):
            ref={'wt' if task=='T3' else 'reference':paths['reference']}
            return veckit.score(task,paths['input'],paths['target'],**ref)['metrics']
        records=[]
        for round_no in range(rounds):
            barrier=threading.Barrier(workers,timeout=30)
            def job(i):
                task=('T1','T2','T3')[i%3]
                try:
                    barrier.wait()
                    return {'task':task,'ok':True,'metrics':score(task)}
                except Exception as exc:
                    return {'task':task,'ok':False,'error':type(exc).__name__+': '+str(exc).replace(str(source.resolve()),'<source>')}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                records.append({'round':round_no+1,'results':list(pool.map(job,range(calls)))})
        serial={task:score(task) for task in ('T1','T2','T3')}
        mismatches=[]
        for record in records:
            for i,row in enumerate(record['results']):
                if row['ok'] and not numeric_equal(row['metrics'],serial[row['task']]):mismatches.append([record['round'],i,row['task']])
        unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest()==before[role] for role,path in paths.items())
    return {'scope':'real public Python API calls on synthetic data; thread failure counts depend on scheduling','python':platform.python_version(),'platform':platform.platform(),'calls_per_round':calls,'workers':workers,'rounds':records,'failed_calls':sum(not row['ok'] for record in records for row in record['results']),'serial_comparison_mismatches':mismatches,'serial_metrics':serial,'module_aliases_preserved':{name:sys.modules.get(name) is module for name,module in sentinels.items()},'sys_path_unchanged':sys.path==path_before,'input_files_unchanged':unchanged}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--calls',type=int,default=12);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--expect',choices=['isolated','collision']);ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.calls<1 or args.workers<1 or args.rounds<1 or args.calls%args.workers:ap.error('positive counts required; calls must be a multiple of workers')
    result=run(args.source,args.calls,args.workers,args.rounds)
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    print(text)
    if args.expect=='isolated':assert result['failed_calls']==0 and not result['serial_comparison_mismatches'] and all(result['module_aliases_preserved'].values()) and result['sys_path_unchanged']
    if args.expect=='collision':assert result['failed_calls']>0 or not all(result['module_aliases_preserved'].values())
    assert result['input_files_unchanged']

if __name__=='__main__':main()
