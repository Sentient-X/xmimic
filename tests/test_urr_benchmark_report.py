"""Paired comparisons preserve grouping and expose regressions per capability."""
import numpy as np
import pytest
from tools.report_urr_benchmark import paired_interval


def test_paired_interval_preserves_known_per_task_difference():
    baseline=np.array([[1.,10.,30.],[3.,8.,40.],[7.,4.,20.]])
    result=paired_interval(baseline+.25,baseline)
    assert result['mean_candidate_minus_baseline']==pytest.approx(.25)
    assert result['upper_bound']==pytest.approx(.25)
    assert result['training_seeds']==3 and result['task_families']==3
    assert 'descriptive' in result['method']


def test_pairing_rejects_incommensurate_or_nonfinite_observations():
    for a,b in (([[1]],[[1,2]]),([[np.nan]],[[0]]),([1,2],[2,3])):
        with pytest.raises(ValueError):
            paired_interval(a,b)
