from typing import Any
import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from diffusers import DDPMScheduler
from define_model import MyUNetClassConditionedModel
import modal
import utils
from tqdm import tqdm
import time
import numpy as np
from torch.profiler import profile, ProfilerActivity
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
import onnxruntime


# with compile:
# mean: 0.009175559878349305
# std: 0.0008318731976560617
# max: 0.01125192642211914

# without compile
# mean: 0.00486307293176651
# std: 0.00026538266776540497
# max: 0.0058710575103759766

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb", 'torchao', 'numpy', 'onnxruntime'
).add_local_python_source(
    "define_model", "get_data", "config", "utils"
).add_local_file(
    # "saved_model.pt", remote_path="/root/saved_model.pt"
    "model.onnx", remote_path="/root/model.onnx"
).add_local_file(
    "model.onnx.data", remote_path="/root/model.onnx.data"
).add_local_file(
    "saved_model.pt", remote_path="/root/saved_model.pt"
)
app = modal.App("mastering-pytorch-my_ddpm")




def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    # device = 'cpu'
    print(f'--- DEVICE: {device} ---')

    noise_scheduler: DDPMScheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps, beta_schedule='squaredcos_cap_v2'
    )
    example_x, _ = dataset[0]
    ch, height, width = example_x.shape
    # net = MyUNetClassConditionedModel(example_x=example_x).to(device)
    # net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    # net.eval()
    # torch.backends.quantized.engine = 'qnnpack'
    # net = torch.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
    # quantize_(net, Int8DynamicActivationInt8WeightConfig())
    # net = torch.compile(net)
    x = torch.randn((10, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    # adapted_timestamp = torch.tensor(999).expand(x.shape[0]).to(device)
    # torch.onnx.export(net, (x, adapted_timestamp, y), f=config.onnx_path)
    session = onnxruntime.InferenceSession(config.onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    log_periods = []
    # for timestamp in range(200):
    #     adapted_timestamp = torch.tensor(timestamp).expand(x.shape[0]).to(device)
    #     with torch.no_grad():
    #         pred = net(x, adapted_timestamp, y)
    #     x = noise_scheduler.step(pred, timestamp, x).prev_sample
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for timestamp in tqdm(noise_scheduler.timesteps):
        start_period = time.time()
        adapted_timestamp = timestamp.expand(x.shape[0]).to(device)
        with torch.no_grad():
            # with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            # pred = net(x, adapted_timestamp, y)
            # inside the loop — convert each tensor to numpy, build the feed dict
            feed = {
                input_names[0]: x.cpu().numpy(),
                input_names[1]: adapted_timestamp.cpu().numpy(),
                input_names[2]: y.cpu().numpy(),
            }
            outputs = session.run(output_names, feed)  # returns a LIST of numpy arrays
            pred = torch.from_numpy(outputs[0]).to(device)  # convert back to a tensor for noise_scheduler.step
        x = noise_scheduler.step(pred, timestamp, x).prev_sample
        log_periods.append(time.time() - start_period)
        yield {'type': 'img', 'x': x.cpu().detach(), 'timestamp': timestamp.item()}
    # profiling
    # print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=20))
    # prof.export_chrome_trace("trace.json")
    yield {'type': 'final', 'log_periods': log_periods, 'x': x.cpu().detach()}


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
def modal_run_inference():
    for chunk in run_inference():
        yield chunk


@app.local_entrypoint()
def modal_main():
    for chunk in modal_run_inference.remote_gen():
        x: torch.Tensor | Any = chunk['x']
        # if chunk['type'] == 'img':
        #     timestamp = chunk['timestamp']
        #     if timestamp % 50 == 0:
        #         show_x = x.permute(0, 2, 3, 1).reshape(280, 28, 1).numpy()
        #         show_x = (show_x + 1) / 2
        #         utils.plot_image(show_x)
        if chunk['type'] == 'final':
            show_x = x.permute(0, 2, 3, 1).reshape(280, 28, 1).numpy()
            show_x = (show_x + 1) / 2
            utils.plot_image(show_x)
            plt.show()
            log_periods = chunk['log_periods']
            n = len(log_periods)
            print(f'mean: {np.mean(log_periods[200:])}')
            print(f'std: {np.std(log_periods[200:])}')
            print(f'max: {np.max(log_periods[200:])}')
            # plt.plot(list(range(200, n)), log_periods[200:])
    plt.show()


def main():
    for chunk in run_inference():
        if chunk['type'] == 'img':
            timestamp = chunk['timestamp']
            x: torch.Tensor | Any = chunk['x']
            if timestamp % 100 == 0:
                show_x = x.cpu().detach().permute(0, 2, 3, 1).reshape(280, 28, 1).numpy()
                show_x = (show_x + 1) / 2
                utils.plot_image(show_x)
        if chunk['type'] == 'final':
            log_periods = chunk['log_periods']
            print(f'mean: {np.mean(log_periods[200:])}')
            print(f'std: {np.std(log_periods[200:])}')
            print(f'max: {np.max(log_periods[200:])}')
            # plt.close()
            # n = len(log_periods)
            # plt.plot(list(range(200, n)), log_periods[200:])
    plt.show()


if __name__ == '__main__':
    main()
