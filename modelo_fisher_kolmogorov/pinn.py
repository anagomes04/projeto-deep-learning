import numpy as np
from scipy import ndimage

from dataclasses import asdict, dataclass

HEALTHY, EDEMA, CORE = 0, 1, 2

@dataclass
class PINNDataConfig:
    core_interval: tuple = (0.75, 1.0)
    edema_interval: tuple = (0.25, 0.75)
    healthy_interval: tuple = (0.0, 0.25)
    include_necrosis_in_core: bool = True
    max_tumor_obs: int = None
    healthy_obs_ratio: float = 1.0
    near_tumor_mm: float = 15.0
    near_fraction: float = 0.5
    n_collocation: int = 100_000
    n_boundary: int = 20_000
    block_mm: float = 8.0
    val_fraction: float = 0.2
    seed: int = 0


def get_observation_points(seg, brain, cfg, rng):
    classification_voxel = build_class_map(seg, brain, cfg)

    tumor = classification_voxel > HEALTHY
    tumor_sample_size = int(tumor.sum()) if cfg.max_tumor_obs is None else min(cfg.max_tumor_obs, int(tumor.sum()))
    tumorous_voxel_sample = sample_voxels(tumor, tumor_sample_size, rng)

    near_tumor_mask = ndimage.distance_transform_edt(seg == 0) <= cfg.near_tumor_mm

    healthy = classification_voxel == HEALTHY
    healthy_sample_size = int(tumor_sample_size * cfg.healthy_obs_ratio)
    healthy_voxel_sample = sample_voxels(healthy, healthy_sample_size, rng, near_tumor_mask, cfg.near_fraction)

    voxel_sample = np.concatenate([tumorous_voxel_sample, healthy_voxel_sample])
    labels = classification_voxel[tuple(voxel_sample.T)]
    intervals = {CORE: cfg.core_interval, EDEMA: cfg.edema_interval, HEALTHY: cfg.healthy_interval}
    bounds = np.array([intervals[c] for c in labels], np.float32)

    return voxel_sample, labels, bounds

def get_collocation_points(brain, seg, cfg, rng):
    near = ndimage.distance_transform_edt(seg == 0) <= cfg.near_tumor_mm
    vox = sample_voxels(brain, cfg.n_collocation, rng, near, cfg.near_fraction).astype(np.float64)
    vox += rng.uniform(-0.5, 0.5, vox.shape)
    t = rng.uniform(0, 1, len(vox))
    return vox

def get_boundary_points(brain, affine, cfg, rng):
    surface = brain & ~ndimage.binary_erosion(brain)
    vox = sample_voxels(surface, cfg.n_boundary, rng)

    smooth = ndimage.gaussian_filter(brain.astype(np.float32), sigma=2.0)
    grad = np.stack(np.gradient(smooth), axis=-1)[tuple(vox.T)]
    with np.errstate(all="ignore"):
        normal = -grad @ affine[:3, :3].T
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12

    t = rng.uniform(0, 1, len(vox))
    return vox, normal


def build_class_map(seg, brain, cfg):
    cls = np.full(seg.shape, -1, np.int8)
    cls[brain & (seg == 0)] = HEALTHY
    cls[seg == 3] = EDEMA
    cls[seg == 1] = CORE
    cls[seg == 2] = CORE if cfg.include_necrosis_in_core else -1
    return cls

def sample_voxels(mask, n, rng, near=None, near_fraction=0.0):
    pool = np.argwhere(mask)
    n = min(n, len(pool))
    if near is None or near_fraction == 0:
        return pool[rng.choice(len(pool), n, replace=False)]
    near_pool, far_pool = np.argwhere(mask & near), np.argwhere(mask & ~near)
    n_near = min(int(round(n * near_fraction)), len(near_pool))
    n_far = min(n - n_near, len(far_pool))
    return np.concatenate([near_pool[rng.choice(len(near_pool), n_near, replace=False)],
                           far_pool[rng.choice(len(far_pool), n_far, replace=False)]])
