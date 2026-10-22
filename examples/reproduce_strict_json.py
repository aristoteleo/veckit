"""Exercise real CLI/API JSON output using temporary, entirely invented input data."""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys,tempfile,warnings
import anndata as ad
import numpy as np
import pandas as pd


def make_files(folder, mode='normal'):
    rng=np.random.default_rng(42);n,g=96,40
    labels=np.repeat(['invented_a','invented_b','invented_c'],32)
    counts=rng.gamma(2,1,(n,g))
    counts[labels=='invented_b',:10]+=2
    counts[labels=='invented_c',10:20]+=2
    coords=rng.normal(size=(n,3))*[1,2,3]
    paths={}
    for role in ['reference','target','input']:
        x=counts.copy()
        if role!='reference':
            x[:,:12]*=3 if role=='target' else 2.7
            x[:,12:24]*=.2 if role=='target' else .3
        x=np.log1p(10000*x/x.sum(1,keepdims=True)).astype(np.float32)
        if mode=='constant_expression' and role=='input':
            x=np.full((n,g),np.log1p(10000/g),np.float32)
        a=ad.AnnData(x,obs=pd.DataFrame({'celltype':labels},index=[f'cell{i}' for i in range(n)]),var=pd.DataFrame(index=[f'invented_gene{i}' for i in range(g)]))
        a.obsm['spatial_3D']=(coords*(1.1 if role=='input' else 1)).astype(np.float32)
        if mode=='collapsed_coordinates' and role=='input':a.obsm['spatial_3D'][:]=0
        path=folder/f'{role}.h5ad';a.write_h5ad(path);paths[role]=path
    return paths


def strict_loads(text):
    def reject(token):raise ValueError('non-JSON numeric token: '+token)
    return json.loads(text,parse_constant=reject)


def inspect_json(text):
    try:
        parsed=strict_loads(text)
        return {'strict_json':True,'null_metrics':[k for k,v in parsed['metrics'].items() if v is None]}
    except ValueError as exc:
        return {'strict_json':False,'error':str(exc)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--expect',choices=['strict','nonstandard'])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args();source=args.source.resolve();sys.path.insert(0,str(source))
    import veckit
    assert Path(veckit.__file__).resolve()==source/'veckit.py'
    rows=[]
    with tempfile.TemporaryDirectory(prefix='strict-json-') as td,warnings.catch_warnings():
        warnings.simplefilter('ignore');td=Path(td)
        for task,mode in [('T1','normal'),('T1','constant_expression'),('T2','collapsed_coordinates'),('T3','collapsed_coordinates')]:
            paths=make_files(td,mode);before={k:p.read_bytes() for k,p in paths.items()}
            ref='wt' if task=='T3' else 'reference';out=td/'api.json'
            result=veckit.score(task,paths['input'],paths['target'],**{ref:paths['reference']},out=out)
            row={'task':task,'mode':mode,'api':inspect_json(out.read_text()),'raw_nonfinite_metrics':[k for k,v in result['metrics'].items() if isinstance(v,(float,np.floating)) and not np.isfinite(v)]}
            cli=subprocess.run([sys.executable,str(source/'score_h5ad.py'),'--task',task,'--input',str(paths['input']),'--target',str(paths['target']),'--'+ref,str(paths['reference']),'--out',str(td/'cli.json')],capture_output=True,text=True)
            if cli.returncode:raise RuntimeError(cli.stderr)
            row['cli']=inspect_json(cli.stdout);row['cli_exit_code']=cli.returncode
            row['cli_stdout_matches_file']=cli.stdout==(td/'cli.json').read_text()
            row['input_files_unchanged']=all(p.read_bytes()==before[k] for k,p in paths.items())
            rows.append(row)
    result={'scope':'JSON serialization only; raw Python metric values retain their numerical semantics','python':platform.python_version(),'platform':platform.platform(),'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['score_h5ad.py','veckit.py']},'cases':rows}
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    print(text)
    if args.expect=='strict':assert all(row[interface]['strict_json'] for row in rows for interface in ['api','cli'])
    if args.expect=='nonstandard':assert any(not row[interface]['strict_json'] for row in rows for interface in ['api','cli'])
    assert all(row['input_files_unchanged'] and row['cli_stdout_matches_file'] for row in rows)

if __name__=='__main__':main()
