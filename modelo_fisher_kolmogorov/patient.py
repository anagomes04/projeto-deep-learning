from pathlib import Path
import nibabel as nib
import numpy as np

class PatientPath:
    path: Path
    weeks: list[Path]

    def __init__(self, path: Path):
        self.path = path
        self.weeks = [p for p in sorted(path.glob('week-*')) if p.is_dir()]

class CoordinateFrame:

    def __init__(self, affine, brain):
        self.affine = affine
        idx = np.argwhere(brain)
        lo, hi = self.vox_to_mm(idx.min(axis=0)), self.vox_to_mm(idx.max(axis=0))
        self.center = ((lo + hi) / 2)[0]
        self.scale = float(np.max(np.abs(hi - lo)) / 2)

    def vox_to_mm(self, ijk):
        ijk = np.atleast_2d(ijk).astype(np.float64)
        with np.errstate(all="ignore"):
            return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def vox_to_norm(self, ijk):
        return (self.vox_to_mm(ijk) - self.center) / self.scale
