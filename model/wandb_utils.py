import os
import tempfile
from pathlib import Path

import wandb

from .litmodules.encoder_lit_module import DriftingMoleculeGenerator

WANDB_ENTITY = "equivariant-drifting"
WANDB_PROJECT = "col-daniel-tests"
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

# TODO: check wether load and saving models is compatible with riemannian encoder

def load_config(wandb_run_id: str):
    api = wandb.Api()
    run = api.run(f"{WANDB_PATH}/{wandb_run_id}")
    return run.config


def save_and_log_model(
    lit_module: DriftingMoleculeGenerator,
    log_model: bool = True,
    save_model: bool = True,
) -> None:
    """Saves trainable model components and logs them as wandb files."""
    wandb_run_dir = wandb.run.dir
    save_path = f"{wandb_run_dir}/individual_components"
    os.makedirs(save_path, exist_ok=True)

    if save_model:
        lit_module.save_individual_components(save_path)

    if log_model:
        for component in lit_module._SAVE_COMPONENTS:
            print(f"Logging {component} to wandb.")
            wandb.save(f"{save_path}/{component}.pth", base_path=wandb_run_dir)


def _download_component(run, remote_name: str, local_path: Path) -> bool:
    """Download a single component file. Returns False if not found (404), raises on other errors."""
    try:
        run.file(f"individual_components/{remote_name}").download(
            root=str(local_path.parent), replace=True
        )
        (local_path.parent / "individual_components" / remote_name).rename(local_path)
        return True
    except wandb.errors.CommError as e:
        if "404" in str(e):
            return False
        raise


def load_pretrained_generator(
    wandb_run_id: str,
    lit_module: DriftingMoleculeGenerator,
    variant: str = "best",
) -> None:
    """Loads generator weights from a given wandb run id into lit_module.

    variant: "best" or "final" — matches the suffix saved by GeneratorCheckpointCallback.
    The feature extractor is loaded only if its weights were saved (i.e. it was fine-tuned).
    """
    api = wandb.Api()
    run = api.run(f"{WANDB_PATH}/{wandb_run_id}")

    print(
        f"Downloading components from wandb run: {WANDB_PATH}/{wandb_run_id} (variant={variant})"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"  Downloading generator_{variant}.pth")
        _download_component(run, f"generator_{variant}.pth", tmp_path / "generator.pth")

        print(f"  Downloading feature_extractor_{variant}.pth")
        found = _download_component(
            run, f"feature_extractor_{variant}.pth", tmp_path / "feature_extractor.pth"
        )
        if not found:
            print(
                f"  feature_extractor_{variant}.pth not found — was not fine-tuned, skipping."
            )

        lit_module.load_individual_components(tmp_path)
