"""Time gene-mismatch validation on temporary, invented H5AD files (not scoring)."""
from pathlib import Path
import argparse,hashlib,json,platform,statistics,sys,tempfile,time
import anndata as ad
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--genes',type=int,nargs='+',default=[500,2000,8000,32285])
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.repeats<1 or min(args.genes)<1: ap.error('counts must be positive')
    sys.path.insert(0,str(args.source.resolve()))
    import score_h5ad
    source=Path(score_h5ad.__file__)
    if source.resolve()!=args.source.resolve()/'score_h5ad.py': raise RuntimeError('unexpected scorer source')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='gene-validation-') as td:
        for count in args.genes:
            genes=[f'invented_gene_{i}' for i in range(count)]
            a=ad.AnnData(np.ones((2,count),dtype=np.float32),var=pd.DataFrame(index=genes[::-1]))
            path=Path(td)/'input.h5ad';a.write_h5ad(path);before=path.read_bytes()
            elapsed=[];messages=[]
            for _ in range(args.repeats):
                start=time.perf_counter()
                try: score_h5ad._prediction(path,genes,need_coords=False,allow_reorder=False)
                except ValueError as exc: messages.append(str(exc))
                else: raise AssertionError('mismatched order was accepted')
                elapsed.append(time.perf_counter()-start)
            assert path.read_bytes()==before and len(set(messages))==1
            rows.append({'genes':count,'cells':2,'seconds':elapsed,'median_seconds':statistics.median(elapsed),'error':messages[0],'input_unchanged':True})
    result={'scope':'gene mismatch validation, including H5AD read; not the scoring pipeline','python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'scorer_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runs':rows}
    text=json.dumps(result,indent=2)
    if args.out: args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    print(text)

if __name__=='__main__':main()
