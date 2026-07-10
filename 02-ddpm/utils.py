import importlib.util
import os
import random
import sys
from pathlib import Path
import modal
import numpy as np
import torch
from torchvision.utils import save_image


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "torchvision", "numpy")
    .env({"HF_HOME": "/vol/hf"})
)
app = modal.App("mastering-pytorch")
volume = modal.Volume.from_name("mastering-pytorch", create_if_missing=True)


def load_impl(project_dir):
    project_dir = Path(project_dir)
    name = os.environ.get("IMPL", "template")
    spec = importlib.util.spec_from_file_location(
        f"{project_dir.name}_{name}", project_dir / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_grid(x, path, nrow=10):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(x.clamp(-1, 1).add(1).div(2), path, nrow=nrow)
