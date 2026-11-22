import json
import re
from pathlib import Path
import nibabel as nib
import numpy as np

from pinn import (get_observation_points, get_collocation_points,
                  get_boundary_points, PINNDataConfig)
from patient import PatientPath, CoordinateFrame

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data';

CLIP_PERCENTILES = (0.5, 99.5)

_WEEK_RE = re.compile(r'^week-(\d+)(?:-(\d+))?$')

def main():
    cfg = PINNDataConfig()

    patient_paths = [PatientPath(p) for p in sorted(DATA_DIR.glob('Patient-*')) if p.is_dir()]
    out_dir = ROOT / 'processed' / 'real'
    out_dir.mkdir(parents=True, exist_ok=True)

    for patient in patient_paths:
        rng = np.random.default_rng(cfg.seed)

        week_indexer, duration = get_week_indexer(patient.weeks)
        obs = []
        segmentation_list = []

        for week in patient.weeks:
            week_index = week_indexer[week.name]
            seg, brain, features, affine = load_week_data(week)

            if not segmentation_list:
                frame, first_brain = CoordinateFrame(affine, brain), brain

            voxel, labels, bounds = get_observation_points(seg, brain, cfg, rng)
            lower_bound, upper_bound = bounds[:, 0], bounds[:, 1]

            xyzt = np.column_stack([
                frame.vox_to_norm(voxel),
                np.full(len(voxel), week_index)
            ]).astype(np.float32)

            obs.append({
                'xyzt': xyzt,
                'lower': lower_bound,
                'upper': upper_bound,
                'labels': labels,
                'features': features[tuple(voxel.T)].astype(np.float32),
            })
            segmentation_list.append(seg)

        obs = {k: np.concatenate([o[k] for o in obs]) for k in obs[0]}

        union_seg = np.max(np.stack(segmentation_list), axis=0)
        col_voxel = get_collocation_points(first_brain, union_seg, cfg, rng).astype(np.float64)
        col_voxel += rng.uniform(-0.5, 0.5, col_voxel.shape)
        col_t = rng.uniform(0, 1, len(col_voxel))
        col = {
            'xyzt': np.column_stack([frame.vox_to_norm(col_voxel), col_t]).astype(np.float32),
        }

        bnd_voxel, bnd_normal = get_boundary_points(first_brain, frame.affine, cfg, rng)
        bnd_t = rng.uniform(0, 1, len(bnd_voxel))
        bnd = {
            'xyzt': np.column_stack([frame.vox_to_norm(bnd_voxel), bnd_t]).astype(np.float32),
            'normal': bnd_normal.astype(np.float32),
        }

        arrays = {}
        for prefix, points in (('obs', obs), ('col', col), ('bnd', bnd)):
            for k, v in points.items():
                arrays[f'{prefix}_{k}'] = v

        core0 = (segmentation_list[0] == 1) | (segmentation_list[0] == 2)
        seed_guess = frame.vox_to_norm(np.argwhere(core0).mean(axis=0))[0]

        meta = {
            'patient': patient.path.name,
            'weeks': [w.name for w in patient.weeks],
            't_of_week': dict(week_indexer),
            'time_span_weeks': duration,
            'single_timepoint': len(patient.weeks) == 1,
            'space': 'DeepBraTumIA atlas (MNI 1 mm)',
            'grid_shape': list(first_brain.shape),
            'affine': frame.affine.tolist(),
            'center_mm': frame.center.tolist(),
            'scale_mm': frame.scale,
            'seed_guess_norm': seed_guess.tolist(),
            'inputs': ['x', 'y', 'z', 't'],
            'feature_channels': ['t1', 't2', 'ct1', 'flair'],
            'counts': {k: int(len(v)) for k, v in arrays.items() if k.endswith('_xyzt')},
        }

        np.savez_compressed(out_dir / f'{patient.path.name}.npz', **arrays)
        with open(out_dir / f'{patient.path.name}.json', 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def get_week_indexer(weeks):
    names = [w.name for w in weeks]
    parsed = {n: parse_week(n) for n in names}

    if len(names) == 1:
        return {names[0]: 0.0}, 0.0

    sub_counts = {}
    for w, s in parsed.values():
        sub_counts[w] = max(sub_counts.get(w, 0), s)

    positions = {
        n: w + (s / (sub_counts[w] + 1) if sub_counts[w] else 0.0)
        for n, (w, s) in parsed.items()
    }

    lo, hi = min(positions.values()), max(positions.values())
    span = hi - lo
    return {n: (p - lo) / span for n, p in positions.items()}, float(span)

def parse_week(name):
    m = _WEEK_RE.match(name)
    if not m:
        raise ValueError(f"unexpected week name: {name}")
    return int(m.group(1)), int(m.group(2) or 0)

def load_week_data(week):
    atlas_path = week / 'DeepBraTumIA-segmentation' / 'atlas'

    images = [
        load_nib_file(str(atlas_path / 'skull_strip' / f'{image}_skull_strip.nii.gz'))
        for image in ['t1', 't2', 'ct1', 'flair']
    ]

    brain, _ = load_nib_file(str(atlas_path / 'skull_strip' / f'brain_mask.nii.gz'))
    brain = brain > 0

    features = np.stack([normalize(data, brain)[0] for data, img in images], axis=-1)

    seg, img = load_nib_file(atlas_path / 'segmentation' / 'seg_mask.nii.gz')

    return seg, brain, features, img.affine

def load_nib_file(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj).astype(np.float32), img

def normalize(raw, brain_mask):
    brain = (brain_mask > 0) & (raw > 0)

    lo, hi = np.percentile(raw[brain], CLIP_PERCENTILES)
    clipped = np.clip(raw, lo, hi)

    mean, std = clipped[brain].mean(), clipped[brain].std()
    norm = np.where(brain, (clipped - mean) / std, 0).astype(np.float32)

    return norm, brain, (lo, hi, mean, std)

if __name__ == '__main__':
    main()
