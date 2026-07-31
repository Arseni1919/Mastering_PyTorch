from typing import Any
import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from define_model import DiTFullModel
from tqdm import tqdm
import onnxruntime


def plot_image(image, timestamp, model_config, pause_time=0.01):
    plt.cla()
    plt.imshow(image)
    plt.title(
        f'timestamp: {timestamp + 1}\n'
        f'hidden_size={model_config.hidden_size} | '
        f'num_layers={model_config.num_layers} | '
        f'n_epochs={model_config.n_epochs}'
    )
    plt.pause(pause_time)


def run_inference():
    device = 'cpu'
    step_size = 1 / config.num_inference_steps
    example_x, _ = dataset[0]
    bs = 10
    ch, height, width = example_x.shape
    x = torch.randn((bs, ch, height, width), device=device)
    y = torch.arange(0, 10, device=device).long()
    t = 0.0

    session = onnxruntime.InferenceSession(config.onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    for _ in tqdm(range(config.num_inference_steps)):
        t_tensor = torch.ones((bs,)) * t
        with torch.no_grad():
            feed = {
                input_names[0]: x.cpu().numpy(),
                input_names[1]: t_tensor.cpu().numpy(),
                input_names[2]: y.cpu().numpy(),
            }
            outputs = session.run(output_names, feed)  # returns a LIST of numpy arrays
            pred = torch.from_numpy(outputs[0]).to(device)
        x += step_size * pred
        t += step_size
        yield {'type': 'img', 'x': x.cpu(), 'timestamp': t}


def main():
    x_list = []
    for chunk in run_inference():
        if chunk['type'] == 'img':
            x: torch.Tensor | Any = chunk['x']
            show_x = x.cpu().detach().permute(0, 2, 3, 1).reshape(10*config.side_size, config.side_size, 1).numpy()
            show_x = (show_x + 1) / 2
            x_list.append(show_x)
    print('Start to show...')
    for t, show_x in enumerate(x_list):
        plot_image(show_x, t, config, pause_time=0.05)
    plt.show()


if __name__ == '__main__':
    main()