import modal
import torch


image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib"
).add_local_python_source("model", "my_dataset", "utils")
app = modal.App("mastering-pytorch-ddpm")


def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount