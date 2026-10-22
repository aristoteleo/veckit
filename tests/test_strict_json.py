"""Strict reports preserve finite values and the Python API's numeric return contract."""
from pathlib import Path
import copy,json,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import score_h5ad as scorer
import veckit
from examples.reproduce_strict_json import make_files,strict_loads


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),np.float32('nan'),np.float32('inf'),np.float64('-inf')])
def test_nonfinite_numbers_become_null_without_mutating_input(value):
    result={'meta':{'label':'NaN / Infinity is text'},'metrics':{'value':value,'nested':[value,{'value':value}],'tuple':(value,)}}
    encoded=scorer._result_json(result);decoded=strict_loads(encoded)
    assert decoded['metrics']=={'value':None,'nested':[None,{'value':None}],'tuple':[None]}
    assert decoded['meta']==result['meta']
    assert result['metrics']['value'] is value
    assert result['metrics']['nested'][0] is value


def test_finite_json_is_byte_identical_to_previous_serialization():
    result={'meta':{'task':'T1','description':'合成示例'},'metrics':{'negative':-.012345,'zero':-0.,'one':1.,'none':None,'count':2,'true':True}}
    before=copy.deepcopy(result)
    assert scorer._result_json(result)==json.dumps(result,indent=2,ensure_ascii=False,default=str)
    assert result==before


def test_python_api_keeps_raw_nan_but_writes_null(tmp_path,monkeypatch):
    scores={'undefined':float('nan'),'positive_inf':float('inf'),'negative_inf':float('-inf'),'finite':.25}
    monkeypatch.setattr(scorer,'score_t1',lambda args:({'task':'T1'},scores))
    out=tmp_path/'nested'/'results.json'
    actual=veckit.score('T1','fake-input.h5ad','fake-target.h5ad',out=out)
    assert actual['metrics'] is scores and np.isnan(actual['metrics']['undefined'])
    assert strict_loads(out.read_text())['metrics']=={'undefined':None,'positive_inf':None,'negative_inf':None,'finite':.25}


def test_cli_uses_same_strict_encoder_for_stdout_and_file(tmp_path,monkeypatch,capsys):
    out=tmp_path/'results.json'
    monkeypatch.setattr(scorer,'score_t1',lambda args:({'task':'T1'},{'undefined':float('nan'),'finite':.25}))
    monkeypatch.setattr(sys,'argv',['veckit','--task','T1','--input','fake.h5ad','--target','target.h5ad','--out',str(out)])
    scorer.main();stdout=capsys.readouterr().out
    assert stdout==out.read_text()
    assert strict_loads(stdout)['metrics']=={'undefined':None,'finite':.25}


@pytest.mark.parametrize('task,mode,undefined',[('T1','constant_expression','pseudobulk_pearson'),('T2','collapsed_coordinates','scale_log_ratio'),('T3','collapsed_coordinates','scale_log_ratio')])
def test_real_metric_edge_cases_remain_machine_readable(task,mode,undefined,tmp_path):
    paths=make_files(tmp_path,mode);before={k:p.read_bytes() for k,p in paths.items()}
    reference={'wt' if task=='T3' else 'reference':paths['reference']}
    out=tmp_path/'report.json';actual=veckit.score(task,paths['input'],paths['target'],**reference,out=out)
    assert np.isnan(actual['metrics'][undefined])
    assert strict_loads(out.read_text())['metrics'][undefined] is None
    assert all(p.read_bytes()==before[k] for k,p in paths.items())


@pytest.mark.parametrize('task',['T1','T2','T3'])
def test_real_finite_metrics_are_unchanged_by_output(task,tmp_path):
    paths=make_files(tmp_path);reference={'wt' if task=='T3' else 'reference':paths['reference']}
    expected=veckit.score(task,paths['input'],paths['target'],**reference)
    assert all(v is None or np.isfinite(v) for v in expected['metrics'].values())
    out=tmp_path/'report.json';actual=veckit.score(task,paths['input'],paths['target'],**reference,out=out)
    assert actual==expected
    assert strict_loads(out.read_text())==expected
