import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co


def create_curr_state(side):
    curr_state = torch.zeros((side, side))
    x = random.randint(0, side - 1)
    y = random.randint(0, side - 1)
    curr_state[x, y] = 1.
    return curr_state.unsqueeze(0)


def create_next_state(curr_state: torch.Tensor, action: int):
    next_state = torch.zeros_like(curr_state)
    if action == 0:  # stay
        return curr_state
    if action == 1: # forward
        next_state[:-1] = curr_state[1:]
        next_state[0] += curr_state[0]
        return next_state
    if action == 2:  # backward
        next_state[1:] = curr_state[:-1]
        next_state[-1] += curr_state[-1]
        return next_state
    if action == 3:  # right
        next_state[:, 1:] = curr_state[:, :-1]  # each column takes the content of the column to its left
        next_state[:, -1] += curr_state[:, -1]  # last column would be dropped; keep it in place
        return next_state
    if action == 4:  # left
        next_state[:, :-1] = curr_state[:, 1:]  # each column takes the content of the column to its left
        next_state[:, 0] += curr_state[:, 0]  # last column would be dropped; keep it in place
        return next_state
    else:
        raise RuntimeError()


def get_nav_data(N: int = 10000, side: int = 32):
    data = []
    for _ in range(N):
        curr_state = create_curr_state(side)
        rand_action = random.randint(0, 4)
        next_state = create_next_state(curr_state, rand_action)
        data.append((curr_state, torch.tensor(rand_action), next_state))
    return data


class NavDataset(Dataset):
    def __init__(self, N: int = 10000, side: int = 32):
        super().__init__()
        self.data = get_nav_data(N, side)

    def __getitem__(self, index) -> _T_co:
        curr_state, rand_action, next_state = self.data[index]
        return curr_state, rand_action, next_state

    def __len__(self):
        return len(self.data)



def main():
    data = get_nav_data(side=10)
    actions_dict = {
        0: 'stay', 1: 'up', 2: 'down', 3: 'right', 4: 'left'
    }
    counter = torch.zeros_like(data[0][0])
    actions_counter = {k: 0 for k in actions_dict.keys()}
    action_labels = [actions_dict[k] for k in actions_counter.keys()]
    fig, ax = plt.subplots(1, 4)
    for i, (curr_state, rand_action, next_state) in enumerate(data):
        rand_action = rand_action.item()
        ax[0].cla()
        ax[0].imshow(curr_state.numpy())
        ax[1].cla()
        ax[1].imshow(next_state.numpy())
        ax[1].set_title(f'action: {actions_dict[rand_action]}')
        counter = counter + curr_state
        ax[2].cla()
        ax[2].imshow(counter.numpy())
        ax[2].set_title(f'count: {i}')
        ax[3].cla()
        actions_counter[rand_action] += 1
        action_vals = [v for k, v in actions_counter.items()]
        ax[3].bar(action_labels, action_vals, color="steelblue")
        ax[3].set_title(f'actions')
        plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    main()