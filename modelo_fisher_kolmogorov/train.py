import sys
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@dataclass
class TrainConfig:
    hidden_width: int = 128
    hidden_depth: int = 3
    model_lr: float = 1e-3
    param_lr: float = 1e-2
    n_epochs: int = 10_000
    data_weight: float = 1.0
    residual_weight: float = 1.0
    boundary_weight: float = 1.0
    log_every: int = 500
    seed: int = 0
    device: str = None
    d_init: float = 1e-3
    rho_init: float = 1.0

    def __post_init__(self):
        if self.device is None:
            self.device = default_device()


class PINN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        layers = [nn.Linear(4, cfg.hidden_width), nn.Tanh()]
        for _ in range(cfg.hidden_depth - 1):
            layers += [nn.Linear(cfg.hidden_width, cfg.hidden_width), nn.Tanh()]
        layers.append(nn.Linear(cfg.hidden_width, 1))
        self.net = nn.Sequential(*layers)
        self.log_d = nn.Parameter(torch.tensor(float(np.log(cfg.d_init))))
        self.log_rho = nn.Parameter(torch.tensor(float(np.log(cfg.rho_init))))

    def forward(self, xyzt):
        return self.net(xyzt).squeeze(-1)

    @property
    def D(self):
        return torch.exp(self.log_d)

    @property
    def rho(self):
        return torch.exp(self.log_rho)


def load_dataset(npz_path, device):
    npz_path = Path(npz_path)
    with np.load(npz_path) as npz:
        arrays = dict(npz)
    with open(npz_path.with_suffix('.json')) as f:
        meta = json.load(f)

    tensors = {}
    for k, v in arrays.items():
        t = torch.as_tensor(np.asarray(v), device=device)
        if (k.startswith('col_') or k.startswith('bnd_')) and k.endswith('_xyzt'):
            t = t.clone().requires_grad_(True)
        tensors[k] = t
    return tensors, meta


def data_loss(model, data):
    u_pred = model(data['obs_xyzt'])
    if 'obs_target' in data:
        return torch.mean((u_pred - data['obs_target']) ** 2)
    lower, upper = data['obs_lower'], data['obs_upper']
    return torch.mean(torch.relu(lower - u_pred) ** 2
                      + torch.relu(u_pred - upper) ** 2)


def residual_loss(model, data):
    xyzt = data['col_xyzt']
    u = model(xyzt)
    grads = torch.autograd.grad(u, xyzt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_z, u_t = grads.unbind(dim=1)
    u_xx = torch.autograd.grad(u_x, xyzt, torch.ones_like(u_x), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, xyzt, torch.ones_like(u_y), create_graph=True)[0][:, 1]
    u_zz = torch.autograd.grad(u_z, xyzt, torch.ones_like(u_z), create_graph=True)[0][:, 2]
    laplacian = u_xx + u_yy + u_zz
    residual = u_t - model.D * laplacian - model.rho * u * (1 - u)
    return torch.mean(residual ** 2)


def boundary_loss(model, data):
    xyzt = data['bnd_xyzt']
    u = model(xyzt)
    grads = torch.autograd.grad(u, xyzt, torch.ones_like(u), create_graph=True)[0]
    flux = (grads[:, :3] * data['bnd_normal']).sum(dim=1)
    return torch.mean(flux ** 2)


def report_recovery(model, meta):
    d_learned = model.D.item()
    rho_learned = model.rho.item()
    print(f"Learned D_norm   = {d_learned:.6g}")
    print(f"Learned rho_norm = {rho_learned:.6g}")

    gt = meta.get('ground_truth')
    if gt:
        d_true = gt['diffusion_norm']
        r_true = gt['proliferation_norm']
        print(f"True    D_norm   = {d_true:.6g}  (abs err {abs(d_true - d_learned):.3g})")
        print(f"True    rho_norm = {r_true:.6g}  (abs err {abs(r_true - rho_learned):.3g})")

    scale = meta.get('scale_mm')
    for span_key, unit in (('time_span_weeks', 'week'), ('time_span_days', 'day')):
        span = meta.get(span_key)
        if scale is not None and span:
            d_phys = d_learned * scale ** 2 / span
            rho_phys = rho_learned / span
            print(f"Learned D_phys   = {d_phys:.4g} mm^2/{unit}")
            print(f"Learned rho_phys = {rho_phys:.4g} /{unit}")
            break


def train(cfg, data, meta):
    torch.manual_seed(cfg.seed)
    model = PINN(cfg).to(cfg.device)
    optimizer = torch.optim.Adam([
        {'params': model.net.parameters(), 'lr': cfg.model_lr},
        {'params': [model.log_d, model.log_rho], 'lr': cfg.param_lr},
    ])

    history = {'loss': [], 'data': [], 'residual': [], 'boundary': [],
               'D_norm': [], 'rho_norm': []}

    for epoch in range(cfg.n_epochs):
        optimizer.zero_grad()
        l_data = data_loss(model, data)
        l_res = residual_loss(model, data)
        l_bnd = boundary_loss(model, data)
        loss = (cfg.data_weight * l_data
                + cfg.residual_weight * l_res
                + cfg.boundary_weight * l_bnd)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['data'].append(l_data.item())
        history['residual'].append(l_res.item())
        history['boundary'].append(l_bnd.item())
        history['D_norm'].append(model.D.item())
        history['rho_norm'].append(model.rho.item())

        if epoch % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
            print(f"ep {epoch:5d}  loss={loss.item():.4e}  "
                  f"data={l_data.item():.4e}  res={l_res.item():.4e}  "
                  f"bnd={l_bnd.item():.4e}  "
                  f"D={model.D.item():.4e}  rho={model.rho.item():.4e}")

    report_recovery(model, meta)
    return model, history


def train_one(npz_path, cfg):
    data, meta = load_dataset(npz_path, cfg.device)
    print(f"Loaded {npz_path.name}: "
          f"obs={len(data['obs_xyzt'])}, "
          f"col={len(data['col_xyzt'])}, "
          f"bnd={len(data['bnd_xyzt'])}")
    print(f"Data loss mode: "
          f"{'MSE on obs_target' if 'obs_target' in data else 'interval on obs_lower/upper'}")

    model, history = train(cfg, data, meta)

    out_path = npz_path.with_name(f"{npz_path.stem}_trained.pt")
    torch.save({'model_state': model.state_dict(),
                'history': history,
                'config': asdict(cfg)}, out_path)
    print(f"Saved to {out_path}")


def collect_datasets(path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob('*.npz')
                      if p.with_suffix('.json').is_file()
                      and not p.stem.endswith('_trained'))
    raise FileNotFoundError(f"Not a file or directory: {path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <path-to-npz-or-directory>")
        return

    dataset_path = Path(sys.argv[1])
    cfg = TrainConfig()
    datasets = collect_datasets(dataset_path)
    if not datasets:
        print(f"No datasets found in {dataset_path}")
        return

    print(f"Device: {cfg.device}  |  datasets to train: {len(datasets)}")
    for i, npz_path in enumerate(datasets, 1):
        print(f"\n===== [{i}/{len(datasets)}] {npz_path.name} =====")
        train_one(npz_path, cfg)


if __name__ == '__main__':
    main()
