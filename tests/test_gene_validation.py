"""Gene diagnostics and explicit reordering retain their existing public behavior."""
from pathlib import Path
import sys
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_h5ad import _prediction


def write_input(tmp_path, genes, storage):
    x=np.arange(1, 3*len(genes)+1,dtype=np.float32).reshape(3,len(genes))
    a=ad.AnnData(storage(x),obs=pd.DataFrame({'celltype':['a','b','a']},index=['c1','c2','c3']),var=pd.DataFrame({'marker':range(len(genes))},index=genes))
    a.obsm['spatial_3D']=np.arange(9,dtype=np.float32).reshape(3,3)
    path=tmp_path/'input.h5ad';a.write_h5ad(path)
    return path,x


@pytest.mark.parametrize('storage',[np.asarray,sparse.csr_matrix,sparse.csc_matrix])
@pytest.mark.parametrize('allow_reorder',[False,True])
def test_exact_panel_is_unchanged(tmp_path,storage,allow_reorder):
    path,x=write_input(tmp_path,['a','b','c'],storage)
    before=path.read_bytes()
    actual,coords,labels,a=_prediction(path,['a','b','c'],need_coords=True,allow_reorder=allow_reorder)
    np.testing.assert_array_equal(actual,x)
    np.testing.assert_array_equal(coords,np.arange(9).reshape(3,3))
    assert labels.tolist()==['a','b','a']
    assert path.read_bytes()==before


@pytest.mark.parametrize('storage',[np.asarray,sparse.csr_matrix,sparse.csc_matrix])
def test_explicit_reorder_aligns_expression_and_gene_metadata(tmp_path,storage):
    path,x=write_input(tmp_path,['c','a','b'],storage)
    before=path.read_bytes()
    actual,_,_,a=_prediction(path,['a','b','c'],need_coords=False,allow_reorder=True)
    np.testing.assert_array_equal(actual,x[:,[1,2,0]])
    assert a.var_names.tolist()==['a','b','c']
    assert a.var['marker'].tolist()==[1,2,0]
    assert a.obs_names.tolist()==['c1','c2','c3']
    assert path.read_bytes()==before


@pytest.mark.parametrize('allow_reorder',[False,True])
def test_missing_and_extra_names_follow_input_order_and_stop_at_ten(tmp_path,allow_reorder):
    expected=[f'missing_{i}' for i in range(13)]+['shared']
    received=['shared']+[f'extra_{i}' for i in range(13)]
    path,_=write_input(tmp_path,received,np.asarray)
    with pytest.raises(ValueError) as exc:
        _prediction(path,expected,need_coords=False,allow_reorder=allow_reorder)
    message=str(exc.value)
    assert f'missing head={expected[:10]}' in message
    assert f'extra head={received[1:11]}' in message
    assert 'missing_10' not in message and 'extra_10' not in message
    assert 'expected 14 genes, got 14' in message


def test_reordered_panel_is_rejected_without_opt_in(tmp_path):
    path,_=write_input(tmp_path,['c','a','b'],np.asarray)
    with pytest.raises(ValueError,match=r'missing head=\[\], extra head=\[\]'):
        _prediction(path,['a','b','c'],need_coords=False,allow_reorder=False)


@pytest.mark.parametrize('expected,received',[(['a','b','c'],['a','b']),(['a','b'],['a','b','c'])])
def test_length_mismatch_is_not_silently_repaired(tmp_path,expected,received):
    path,_=write_input(tmp_path,received,np.asarray)
    with pytest.raises(ValueError,match='do not match'):
        _prediction(path,expected,need_coords=False,allow_reorder=True)


def test_reference_bootstraps_panel_without_a_fixed_gene_count(tmp_path):
    path,x=write_input(tmp_path,['a','b','c'],np.asarray)
    actual,_,_,a=_prediction(path,None,need_coords=False,allow_reorder=False)
    np.testing.assert_array_equal(actual,x)
    assert a.var_names.tolist()==['a','b','c']
