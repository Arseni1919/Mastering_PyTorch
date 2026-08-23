import torch
from define_model import MLP
import matplotlib.pyplot as plt
import numpy as np
from config import config


def run_inference():

    N = 100

    model = MLP(layers=config.layers, channels=config.channels)
    model.load_state_dict(torch.load('state_dict.pt'))

    fig, ax = plt.subplots(figsize=(7, 7))

    xt = torch.randn((N, 2))
    xt_list = []
    pred_list = []
    for i, t in enumerate(torch.linspace(0, 1, config.steps), start=1):
        print(f'\r{i}/{config.steps}', end='')
        pred = model(xt, t.expand(xt.shape[0]))
        xt = xt + (1 / config.steps) * pred
        xt_list.append(xt)
        pred_list.append(pred)

        x = [i[0].item() for i in xt]
        y = [i[1].item() for i in xt]
        ax.cla()
        ax.scatter(x, y)
        ax.set_title(f'{i}/{config.steps}')
        plt.pause(0.001)
    plt.show()

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    t_list = list(range(len(xt_list)))
    for i_n in range(N):
        x_list = [i_xt[i_n][0].item() for i_xt in xt_list]
        y_list = [i_xt[i_n][1].item() for i_xt in xt_list]
        ax.plot(x_list, y_list, t_list, color='blue', linewidth=2, alpha=0.1)
        ax.plot(
            [x_list[0], pred_list[0][i_n][0].item()],
            [y_list[0], pred_list[0][i_n][1].item()],
            [t_list[0], t_list[-1]],
            color='red', linewidth=2, alpha=0.1
        )

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('t')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    run_inference()