"""Dense recorded-force previews retain action timing and real camera geometry."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def videos():
    path = Path(__file__).resolve().parents[1] / 'scripts/dataset_videos/render.py'
    spec = importlib.util.spec_from_file_location('dataset_video_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outgoing_interval_force_and_terminal(videos):
    forces = np.array([1.25, 1.25, -2.5])
    assert [videos.current_force(forces, k) for k in range(4)] == [1.25, 1.25, -2.5, None]
    with pytest.raises(IndexError): videos.current_force(forces, 4)


def test_selection_covers_each_collection_family_and_truncated_example(videos):
    manifest = {'schema_version': 3, 'dataset': 'controlled', 'camera': {}, 'train': []}
    for family in ('pulse', 'multisine', 'noisy_lqr', 'lqr_release'):
        for i in range(3):
            manifest['train'].append({'path': f'train/{family}{i}.npz', 'truth': f'truth/{family}{i}.npz',
                'apparatus_id': 'opaque', 'collection_family': family, 'termination_reason': 'duration'})
    manifest['train'][-1]['termination_reason'] = 'boundary_exit'
    selected = videos.select_clips(manifest)
    assert len(selected) == 9
    assert selected == videos.select_clips(manifest)
    assert sum(row['kind'] == 'lqr_release' for row in selected) == 3
    assert len(videos.select_clips(manifest, all_clips=True)) == 12


def test_loader_uses_dense_recorded_arrays_without_mutation(tmp_path, videos):
    rgb = np.full((5, 96, 96, 3), 242, dtype=np.uint8)
    force = np.array([[1.], [1.], [-2.], [-2.]])
    times = np.arange(5, dtype=np.float64) * .01
    state = np.array([[p, 999., 0., 0.] for p in (-.1, -.05, 0., .05, .1)])
    np.savez(tmp_path/'learning.npz', rgb=rgb, force=force, t=times, reset_mode='nonlinear')
    np.savez(tmp_path/'truth.npz', state=state)
    record = {'path': 'learning.npz', 'truth_path': 'truth.npz', 'split': 'train', 'kind': 'pulse',
              'apparatus_id': 'opaque', 'group': None, 'camera': {'x_limits_m': [-2.6, 2.6], 'y_limits_m': [-2.6, 2.6]}}
    before = (tmp_path/'learning.npz').read_bytes()
    clip = videos.load_clip(tmp_path, record)
    np.testing.assert_array_equal(clip['frames'], rgb)
    np.testing.assert_array_equal(clip['forces'], force[:, 0])
    np.testing.assert_array_equal(clip['positions'], state[:, 0])
    assert clip['dt'] == .01 and clip['reset_mode'] == 1
    assert videos.render_frame(clip, 0).size == videos.SIZE
    assert (tmp_path/'learning.npz').read_bytes() == before
