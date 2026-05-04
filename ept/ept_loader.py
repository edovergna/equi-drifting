import subprocess
import sys
from pathlib import Path

import torch

from .ept import EPTFeatureExtractor

_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1tBqGwC_jcTdq3QArFZox_auSCzxDjA0P"
)
_CKPT_RELATIVE = Path("hybrid_noaf") / "epoch49_step215752.ckpt"
_ROOT_PATH = Path(__file__).parents[1]


def load_ept_feature_extractor() -> EPTFeatureExtractor:
    """Locate (downloading if needed) the EPT checkpoint and return an initialised extractor."""
    ckpt_path = _ROOT_PATH / _CKPT_RELATIVE

    if not ckpt_path.exists():
        _download_ept(ckpt_path)

    ept_dir = str(_ROOT_PATH / "ept")
    if ept_dir not in sys.path:
        sys.path.append(ept_dir)

    return EPTFeatureExtractor(str(ckpt_path), torch.device("cpu"))


def _download_ept(expected_path: Path) -> None:
    print(
        f"EPT checkpoint not found at {expected_path}. Downloading from Google Drive..."
    )
    result = subprocess.run(
        ["gdown", "--folder", _GDRIVE_FOLDER],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"EPT download failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            "Ensure gdown is installed: pip install gdown"
        )
    if not expected_path.exists():
        raise RuntimeError(
            f"Download appeared to succeed but {expected_path} not found.\n"
            f"gdown output: {result.stdout}"
        )
    print("Download complete.")
