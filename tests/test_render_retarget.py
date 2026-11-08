import numpy as np
import pytest
from tools.render_retarget import display_indices, limit_hits


def test_display_downsamples_only_and_keeps_both_endpoints():
    frames = display_indices(599, 1 / 0.008333, 30)
    assert len(frames) == 150
    assert frames[0] == 0 and frames[-1] == 598
    assert np.all(np.diff(frames) > 0)
    assert np.max(np.abs(frames[:-1] * 0.008333 - np.arange(149) / 30)) <= 0.008333 / 2


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_display_fps_refused(fps):
    with pytest.raises(ValueError):
        display_indices(599, 120, fps)


def test_limit_display_excludes_declared_holds_and_includes_lower_and_upper():
    q = np.array([[0.0, 1.0, 0.5], [-1.0, 0.0, 0.5]])
    np.testing.assert_array_equal(
        limit_hits(q, [-1, -1, 0.5], [1, 1, 0.5], [True, True, False]),
        [[False, True, False], [True, False, False]],
    )
