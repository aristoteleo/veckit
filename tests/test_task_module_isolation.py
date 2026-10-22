"""Real cold-start concurrent scoring, import isolation and stable public module loading."""
from pathlib import Path
import json,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from score_h5ad import _load_task_metrics


@pytest.mark.parametrize('task',['T1','T2','T3'])
def test_modules_have_task_qualified_names_and_stable_identity(task):
    metrics,panel=_load_task_metrics(task)
    assert metrics.__name__==f'{task}.metrics'
    assert panel.__name__==f'{task}.metrics_v2'
    assert _load_task_metrics(task)==(metrics,panel)
    assert callable(getattr(panel,f'score_task{task[1:]}_v2'))


def test_cold_start_mixed_task_threads_preserve_scores_and_foreign_modules(tmp_path):
    out=tmp_path/'result.json'
    result=subprocess.run([sys.executable,str(ROOT/'examples/reproduce_concurrent_scoring.py'),'--source',str(ROOT),'--rounds','3','--expect','isolated','--out',str(out)],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    data=json.loads(out.read_text())
    assert data['failed_calls']==0
    assert not data['serial_comparison_mismatches']
    assert all(data['module_aliases_preserved'].values())
    assert data['sys_path_unchanged'] and data['input_files_unchanged']
    assert sum(len(r['results']) for r in data['rounds'])==36


def test_no_legacy_reuse_or_task_data_imports():
    code='''
import importlib.abc, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class RejectPrivateLoaders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_reuse", "data", "T1.data", "T2.data", "T3.data"}:
            raise AssertionError("unexpected loader import: " + fullname)
sys.meta_path.insert(0, RejectPrivateLoaders())
from score_h5ad import _load_task_metrics
for task in ("T1", "T2", "T3"):
    metrics, panel = _load_task_metrics(task)
    assert metrics.__name__ == task + ".metrics"
    assert panel.__name__ == task + ".metrics_v2"
'''
    result=subprocess.run([sys.executable,'-c',code,str(ROOT)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_invalid_task_is_rejected_before_import():
    with pytest.raises(ValueError,match='task must be'):_load_task_metrics('invalid')
