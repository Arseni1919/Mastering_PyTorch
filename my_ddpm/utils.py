import modal

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "create_model", "get_data"
).add_local_file(
    "model.pt", remote_path="/root/model.pt"
)
app = modal.App("mastering-pytorch-ddpm_simple")