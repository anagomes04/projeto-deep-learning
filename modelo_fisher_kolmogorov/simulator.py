import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

HEALTHY, EDEMA, CORE = 0, 1, 2


@dataclass
class SimConfig:
    diffusion_mm2_per_day: float = 0.10
    proliferation_per_day: float = 0.05

    scale_mm: float = 90.0
    time_span_days: float = 70.0

    grid_size: int = 32
    n_timesteps: int = 100

    seed_sigma_mm: float = 10.0

    n_obs: int = 50_000
    n_collocation: int = 200_000
    n_boundary: int = 10_000
    seed: int = 0

    core_lo: float = 0.75
    edema_lo: float = 0.25


def build_laplacian(n, dx):
    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n - 1)
    l1 = sparse.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='lil')
    l1[0, 1] = 2
    l1[-1, -2] = 2
    l1 = l1.tocsr() / dx ** 2

    identity = sparse.eye(n, format='csr')
    return (sparse.kron(sparse.kron(l1, identity), identity)
            + sparse.kron(sparse.kron(identity, l1), identity)
            + sparse.kron(sparse.kron(identity, identity), l1)).tocsr()


def simulate(cfg):
    n, t_steps = cfg.grid_size, cfg.n_timesteps
    dx = 2 * cfg.scale_mm / (n - 1)
    dt = cfg.time_span_days / (t_steps - 1)

    axis_mm = np.linspace(-cfg.scale_mm, cfg.scale_mm, n)
    xx, yy, zz = np.meshgrid(axis_mm, axis_mm, axis_mm, indexing='ij')
    r2 = xx ** 2 + yy ** 2 + zz ** 2
    u = np.exp(-r2 / (2 * cfg.seed_sigma_mm ** 2)).reshape(-1)

    lap = build_laplacian(n, dx)
    d, rho = cfg.diffusion_mm2_per_day, cfg.proliferation_per_day
    identity = sparse.eye(n ** 3, format='csr')
    a = (identity - 0.5 * dt * d * lap).tocsc()
    b = (identity + 0.5 * dt * d * lap).tocsr()
    solve = splu(a).solve

    history = np.empty((t_steps, n ** 3), dtype=np.float32)
    history[0] = u
    for k in range(1, t_steps):
        rhs = b @ u + dt * rho * u * (1 - u)
        u = solve(rhs)
        np.clip(u, 0.0, 1.0, out=u)
        history[k] = u

    return history.reshape(t_steps, n, n, n), axis_mm


def sample_cube_boundary(n, rng):
    face = rng.integers(0, 6, size=n)
    tang = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    xyz = np.empty((n, 3), dtype=np.float32)
    normal = np.zeros((n, 3), dtype=np.float32)
    axis_of_face = np.array([0, 0, 1, 1, 2, 2])
    sign_of_face = np.array([1., -1., 1., -1., 1., -1.], dtype=np.float32)
    for f in range(6):
        m = face == f
        if not m.any():
            continue
        ax = axis_of_face[f]
        xyz[m, ax] = sign_of_face[f]
        xyz[m, (ax + 1) % 3] = tang[m, 0]
        xyz[m, (ax + 2) % 3] = tang[m, 1]
        normal[m, ax] = sign_of_face[f]
    t = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    return np.concatenate([xyz, t], axis=1), normal


def to_pinn_arrays(history, axis_mm, cfg):
    rng = np.random.default_rng(cfg.seed)
    t_steps, n = history.shape[0], history.shape[1]

    norm_axis = (axis_mm / cfg.scale_mm).astype(np.float32)
    norm_time = np.linspace(0, 1, t_steps, dtype=np.float32)

    t_idx = rng.integers(0, t_steps, size=cfg.n_obs)
    i_idx = rng.integers(0, n, size=cfg.n_obs)
    j_idx = rng.integers(0, n, size=cfg.n_obs)
    k_idx = rng.integers(0, n, size=cfg.n_obs)
    u_true = history[t_idx, i_idx, j_idx, k_idx].astype(np.float32)
    obs_xyzt = np.stack([norm_axis[i_idx], norm_axis[j_idx], norm_axis[k_idx],
                         norm_time[t_idx]], axis=-1)

    labels = np.where(u_true >= cfg.core_lo, CORE,
             np.where(u_true >= cfg.edema_lo, EDEMA, HEALTHY)).astype(np.int8)
    lower = np.where(labels == CORE, cfg.core_lo,
            np.where(labels == EDEMA, cfg.edema_lo, 0.0)).astype(np.float32)
    upper = np.where(labels == CORE, 1.0,
            np.where(labels == EDEMA, cfg.core_lo, cfg.edema_lo)).astype(np.float32)

    col_xyzt = rng.uniform(low=[-1, -1, -1, 0], high=[1, 1, 1, 1],
                           size=(cfg.n_collocation, 4)).astype(np.float32)
    bnd_xyzt, bnd_normal = sample_cube_boundary(cfg.n_boundary, rng)

    return {
        'obs_xyzt': obs_xyzt, 'obs_target': u_true,
        'obs_lower': lower, 'obs_upper': upper, 'obs_label': labels,
        'col_xyzt': col_xyzt,
        'bnd_xyzt': bnd_xyzt, 'bnd_normal': bnd_normal,
    }


def metadata(cfg):
    d_norm = cfg.diffusion_mm2_per_day * cfg.time_span_days / cfg.scale_mm ** 2
    rho_norm = cfg.proliferation_per_day * cfg.time_span_days
    return {
        'kind': 'synthetic (3D Fisher-Kolmogorov, cube domain, Neumann BC)',
        'space': 'cube [-1, 1]^3, isotropic',
        'grid_shape': [cfg.grid_size] * 3,
        'scale_mm': cfg.scale_mm,
        'time_span_days': cfg.time_span_days,
        'inputs': ['x', 'y', 'z', 't'],
        'classes': {str(HEALTHY): 'healthy', str(EDEMA): 'edema', str(CORE): 'core'},
        'intervals': {'healthy': [0.0, cfg.edema_lo],
                      'edema': [cfg.edema_lo, cfg.core_lo],
                      'core': [cfg.core_lo, 1.0]},
        'ground_truth': {
            'diffusion_mm2_per_day': cfg.diffusion_mm2_per_day,
            'proliferation_per_day': cfg.proliferation_per_day,
            'diffusion_norm': d_norm,
            'proliferation_norm': rho_norm,
        },
        'config': asdict(cfg),
    }


def main():
    cfg = SimConfig()
    out_dir = Path(__file__).parent / 'processed' / 'sim'
    out_dir.mkdir(parents=True, exist_ok=True)

    history, axis_mm = simulate(cfg)

    arrays = to_pinn_arrays(history, axis_mm, cfg)
    meta = metadata(cfg)

    np.savez_compressed(out_dir / 'test_synthetic.npz', **arrays)
    with open(out_dir / 'test_synthetic.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
