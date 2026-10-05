"""Offline dense-corpus QA; hidden states are read only for dataset inspection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from pi_jepa.data import (CART_LIMIT, RASTER_MARGIN_PIXELS, SCHEMA_VERSION,
                          require_visible, validate_manifest, world_to_pixel)
from pi_jepa.physics import ELL, G, M_POLE, lqr_gain, rk4, rollout
from pi_jepa.runs import file_digest, run_directory


def energy(state, theta):
    _, v, q, w = state.unbind(-1)
    mass = theta[..., 0]
    return (.5 * (mass + M_POLE) * v.square() + M_POLE * ELL * v * w * q.cos()
            + .5 * M_POLE * ELL**2 * w.square() - M_POLE * G * ELL * q.cos())


def numerical_checks():
    """Numerical convergence and energy checks, independent of collected episodes."""
    with torch.no_grad():
        state = torch.tensor([[0., .2, .7, -2.], [.3, -1., 3.12, 8.],
                              [-.5, 2., -2.1, -6.]], dtype=torch.float64)
        theta = torch.tensor([[1., .25], [.5, .02], [1.6, .8]], dtype=torch.float64)
        force = torch.tensor([0., 5., -3.], dtype=torch.float64)
        coarse = rk4(state, theta, force, dt=.01, substeps=5)
        reference = rk4(state, theta, force, dt=.01, substeps=10)
        errors = (coarse - reference).abs().amax(0).tolist()
        assert max(errors) < 1e-5, errors
        initial = state[:1].clone()
        conservation = []
        for drag in (0., .25):
            parameters = torch.tensor([[1., drag]], dtype=torch.float64)
            states = rollout(initial, parameters, torch.zeros(1, 300, dtype=torch.float64), dt=.01, substeps=5)
            energies = energy(states, parameters[:, None])
            delta = energies.diff(dim=1)
            conservation.append({'drag': drag, 'initial_joule': float(energies[0, 0]),
                'final_joule': float(energies[0, -1]),
                'maximum_absolute_energy_change_joule': float((energies - energies[:, :1]).abs().max()),
                'maximum_record_energy_increase_joule': float(delta.max())})
        assert conservation[0]['maximum_absolute_energy_change_joule'] < 1e-6
        assert conservation[1]['maximum_record_energy_increase_joule'] < 1e-8
        assert conservation[1]['final_joule'] < conservation[1]['initial_joule']
    return {'rk4_2ms_vs_1ms_max_absolute_error_p_v_q_w': errors,
            'reference_interval_s': .01, 'energy_checks': conservation,
            'nominal_lqr_gain': lqr_gain([1., .25]).reshape(-1).tolist()}


def entries_by_split(manifest):
    splits = ('train', 'validation', 'test') if manifest['dataset'] == 'passive' else ('train', 'validation', 'calibration', 'query')
    return [(split, row) for split in splits for row in manifest.get(split, [])]


def describe(values):
    a = np.concatenate(values).astype(np.float64).reshape(-1)
    if len(a) == 0:
        return {'count': 0}
    return {'count': len(a), 'mean': float(a.mean()), 'std': float(a.std()),
            'percentiles': {str(q): float(v) for q, v in zip((0, 1, 5, 50, 95, 99, 100),
                                      np.percentile(a, (0, 1, 5, 50, 95, 99, 100)))}}


def save_examples(root, selected, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    for number, (split, entry) in enumerate(selected):
        with np.load(root / entry['truth'], allow_pickle=False) as saved:
            state = saved['state'].copy()
        with np.load(root / entry['path'], allow_pickle=False) as saved:
            times, force = saved['t'], saved['force'].reshape(-1)
            rgb = saved['rgb']
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(times, state[:, 0]); axes[0].set_ylabel('cart p (m)')
        axes[0].axhline(CART_LIMIT, color='gray', linestyle=':'); axes[0].axhline(-CART_LIMIT, color='gray', linestyle=':')
        axes[1].plot(times, state[:, 2]); axes[1].set_ylabel('unwrapped q (rad)')
        axes[2].step(times[:-1], force, where='post'); axes[2].set_ylabel('applied force (N)')
        axes[2].set_xlabel('time (s)')
        fig.suptitle(f"{split} / {entry['collection_family']} / {entry['termination_reason']}")
        fig.tight_layout()
        stem = f"example_{number:02d}_{entry['collection_family']}"
        fig.savefig(destination / f'{stem}.png', dpi=120); plt.close(fig)
        # Stored RGB at selected timestamps; no re-rendered/interpolated frames.
        from PIL import Image
        indices = np.linspace(0, len(rgb)-1, 8, dtype=int)
        strip = np.concatenate([rgb[i] for i in indices], axis=1)
        Image.fromarray(strip).save(destination / f'{stem}_frames.png')
        records.append({'path': entry['path'], 'truth': entry['truth'], 'family': entry['collection_family'],
                        'reason': entry['termination_reason'], 'plot': f'{stem}.png',
                        'contact_sheet': f'{stem}_frames.png', 'frame_indices': indices.tolist()})
    return records


def inspect_corpus(root, output_root, numerics):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    with run_directory(output_root, f"dataset-qa-{manifest['dataset']}-seed42") as destination:
        groups, selected, selection_counts = {}, [], defaultdict(int)
        special = None
        pairs = entries_by_split(manifest)
        # Camera values are recorded with the corpus; isotropic scale is checked separately.
        camera = manifest['camera']
        ppm = float(camera['pixels_per_metre'])
        assert np.isclose(ppm, (camera['width'] - 1) / np.ptp(camera['x_limits_m']))
        assert np.isclose(ppm, (camera['height'] - 1) / np.ptp(camera['y_limits_m']))
        minimum_margin = float('inf')
        minimum_margin_episode = None
        maximum_rod_length_error = 0.
        maximum_cart_position = 0.
        for i, (split, entry) in enumerate(pairs):
            with np.load(root / entry['path'], allow_pickle=False) as saved:
                rgb, force, times = saved['rgb'], saved['force'].reshape(-1), saved['t']
            with np.load(root / entry['truth'], allow_pickle=False) as saved:
                state = saved['state']
            n = len(force)
            assert len(rgb) == len(times) == len(state) == n + 1
            assert rgb.shape[1:] == (96, 96, 3) and rgb.dtype == np.uint8
            assert times.dtype == np.float64 and np.allclose(times, np.arange(n+1)*.01, atol=1e-12, rtol=0)
            assert np.array_equal(force[:2*(n//2):2], force[1:2*(n//2):2])
            assert np.isfinite(state).all() and np.isfinite(force).all() and np.max(np.abs(force), initial=0) <= 5
            assert abs(float(state[0, 0])) < 1e-12
            assert np.all(np.abs(state[:-1, 0]) <= CART_LIMIT)
            exits = bool(np.abs(state[-1, 0]) > CART_LIMIT)
            assert exits == (entry['termination_reason'] == 'boundary_exit')
            margin = require_visible(state, entry['episode_id'])
            assert np.isclose(entry['minimum_visibility_margin_pixels'], margin)
            if margin < minimum_margin:
                minimum_margin, minimum_margin_episode = margin, entry['path']
            maximum_cart_position = max(maximum_cart_position, float(np.abs(state[:, 0]).max()))
            if split in ('train', 'validation'): assert len(rgb) >= 65
            if manifest['dataset'] == 'passive': assert np.count_nonzero(force) == 0
            key = f"{split}/{entry['collection_family']}"
            if key not in groups:
                groups[key] = {'episodes': 0, 'transitions': 0, 'boundary_exits': 0,
                    'terminations': defaultdict(int), 'series': defaultdict(list),
                    'saturated_intervals': 0, 'almost_identical_intervals': 0,
                    'identical_intervals': 0, 'angle_histogram_36bins': np.zeros(36, dtype=np.int64)}
            group = groups[key]
            group['episodes'] += 1; group['transitions'] += n
            group['boundary_exits'] += int(exits)
            group['terminations'][entry['termination_reason']] += 1
            for j, name in enumerate(('p_m', 'v_m_per_s', 'q_unwrapped_rad', 'w_rad_per_s')):
                group['series'][name].append(state[:, j])
            group['series']['force_n'].append(force)
            group['series']['duration_s'].append(np.array([times[-1]]))
            group['saturated_intervals'] += int((np.abs(force) >= 5 - 1e-6).sum())
            angle = np.arctan2(np.sin(state[:, 2]), np.cos(state[:, 2]))
            group['angle_histogram_36bins'] += np.histogram(angle, bins=36, range=(-np.pi, np.pi))[0]
            cart_delta = np.abs(np.diff(state[:, 0]))
            bob = np.stack((state[:, 0] + ELL*np.sin(state[:, 2]), -ELL*np.cos(state[:, 2])), axis=-1)
            pivot_pixels = np.asarray(world_to_pixel(state[:, 0], np.zeros(len(state))))
            bob_pixels = np.asarray(world_to_pixel(bob[:, 0], bob[:, 1]))
            rod_error = np.abs(np.linalg.norm(bob_pixels - pivot_pixels, axis=0) - ELL*ppm)
            maximum_rod_length_error = max(maximum_rod_length_error, float(rod_error.max()))
            assert np.max(rod_error) < 1e-10
            bob_delta = np.linalg.norm(np.diff(bob, axis=0), axis=-1)
            for name, values in (('cart_displacement_m', cart_delta), ('bob_displacement_m', bob_delta),
                                 ('cart_displacement_pixels', cart_delta*ppm), ('bob_displacement_pixels', bob_delta*ppm),
                                 ('absolute_angle_increment_rad', np.abs(np.diff(state[:, 2])))):
                group['series'][name].append(values)
            diff = np.abs(rgb[1:].astype(np.int16) - rgb[:-1].astype(np.int16))
            change = diff.mean(axis=(1, 2, 3))
            group['series']['frame_mean_absolute_change_uint8'].append(change)
            group['series']['frame_changed_channel_fraction'].append((diff != 0).mean(axis=(1, 2, 3)))
            group['almost_identical_intervals'] += int((change < .01).sum())
            group['identical_intervals'] += int((change == 0).sum())
            family = entry['collection_family']
            if split == 'train' and selection_counts[family] < 2:
                selected.append((split, entry)); selection_counts[family] += 1
            if special is None and (exits or (np.abs(force) >= 5-1e-6).any()):
                special = (split, entry)
            if (i+1) % 200 == 0: print(f"QA {manifest['dataset']}: {i+1}/{len(pairs)} episodes", flush=True)
        if special and special not in selected: selected.append(special)
        for group in groups.values():
            group['distributions'] = {name: describe(values) for name, values in group.pop('series').items()}
            group['angle_histogram_36bins'] = group['angle_histogram_36bins'].tolist()
            for name in ('saturated', 'almost_identical', 'identical'):
                group[f'{name}_fraction'] = group[f'{name}_intervals'] / max(1, group['transitions'])
            group['boundary_exit_fraction'] = group['boundary_exits'] / group['episodes']
        examples = save_examples(root, selected, destination)
        report = {'dataset': manifest['dataset'], 'manifest_sha256': file_digest(manifest_path),
            'schema_version': SCHEMA_VERSION, 'record_hz': 100, 'physics_hz': 500, 'action_hz': 50,
            'predictor_hz': 10, 'camera': camera, 'numerical_checks': numerics,
            'cart_limit_m': CART_LIMIT,
            'visibility': {'all_stored_frames_visible': True,
                'minimum_edge_clearance_pixels': minimum_margin,
                'minimum_required_edge_clearance_pixels': RASTER_MARGIN_PIXELS,
                'minimum_margin_episode': minimum_margin_episode,
                'maximum_absolute_cart_position_m': maximum_cart_position,
                'rod_length_m': ELL, 'rod_length_pixels': ELL*ppm,
                'maximum_projected_rod_length_error_pixels': maximum_rod_length_error,
                'includes_first_finite_boundary_exit_frame': True},
            'alignment_checks': 'all stored episodes: N+1 observations/N actions, .01s timestamps, paired held forces, finite prefixes',
            'frame_similarity_definition': 'mean absolute RGB uint8 channel change < 0.01 per recorded interval',
            'angle_histogram_edges_rad': np.linspace(-np.pi, np.pi, 37).tolist(),
            'groups': groups, 'examples': examples, 'collection_counts': manifest.get('counts'),
            'learning_status': 'Collection and interface QA only; useful learned dynamics have not been established.'}
        (destination / 'qa.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"QA complete: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('workspace/data'))
    parser.add_argument('--dataset', choices=('passive', 'controlled', 'both'), default='both')
    parser.add_argument('--output-root', type=Path, default=Path('workspace/evaluations'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = numerical_checks()
    for dataset in (('passive', 'controlled') if args.dataset == 'both' else (args.dataset,)):
        inspect_corpus(args.data_root / dataset, args.output_root, checks)


if __name__ == '__main__':
    main()
