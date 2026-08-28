import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co
from tqdm import tqdm
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def create_curr_state(side):
    curr_state = torch.zeros((side, side))
    padding = 2
    pos_is_ok = False
    while not pos_is_ok:
        x = random.randint(0, side - 1)
        y = random.randint(0, side - 1)
        up_limit = side - padding
        if padding <= x < up_limit and padding <= y < up_limit:
            continue
        curr_state[x, y] = 1.
        pos_is_ok = True
    return curr_state.unsqueeze(0)


def create_next_state(curr_state: torch.Tensor, action: int):
    next_state = torch.zeros_like(curr_state)
    if action == 0:  # stay
        return curr_state
    # NOTE: curr_state is (C, H, W) -- slice the last two dims, not dims 0 and 1
    if action == 1: # forward
        next_state[..., :-1, :] = curr_state[..., 1:, :]  # each row takes the content of the row below it
        next_state[..., 0, :] += curr_state[..., 0, :]  # top row would be dropped; keep it in place
        return next_state
    if action == 2:  # backward
        next_state[..., 1:, :] = curr_state[..., :-1, :]  # each row takes the content of the row above it
        next_state[..., -1, :] += curr_state[..., -1, :]  # bottom row would be dropped; keep it in place
        return next_state
    if action == 3:  # right
        next_state[..., :, 1:] = curr_state[..., :, :-1]  # each column takes the content of the column to its left
        next_state[..., :, -1] += curr_state[..., :, -1]  # last column would be dropped; keep it in place
        return next_state
    if action == 4:  # left
        next_state[..., :, :-1] = curr_state[..., :, 1:]  # each column takes the content of the column to its right
        next_state[..., :, 0] += curr_state[..., :, 0]  # first column would be dropped; keep it in place
        return next_state
    else:
        raise RuntimeError()


def get_nav_data(N: int = 10000, side: int = 32):
    data = []
    for _ in tqdm(range(N)):
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


class PointMazeDataset(Dataset):
    def __init__(self, N: int = 10000):
        super().__init__()
        self.data = []
        env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100)
        # env = gym.make('PointMaze_Large-v3', max_episode_steps=100)
        # env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100)

        curr_obs, info = env.reset(seed=42)
        curr_obs = torch.tensor(curr_obs['observation'])
        for i in tqdm(range(N)):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                curr_obs, info = env.reset()
                curr_obs = torch.tensor(curr_obs['observation'])
                continue

            action = torch.tensor(action)
            next_obs = torch.tensor(next_obs['observation'])
            self.data.append([curr_obs, action, next_obs])
            if not isinstance(curr_obs, torch.Tensor) or not isinstance(next_obs, torch.Tensor) or not isinstance(action, torch.Tensor):
                raise RuntimeError('nope')
            curr_obs = next_obs
        env.close()

    def __getitem__(self, index) -> _T_co:
        curr_state, rand_action, next_state = self.data[index]
        return curr_state, rand_action, next_state

    def __len__(self):
        return len(self.data)



def main():
    data = get_nav_data(side=15)
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
        ax[0].imshow(curr_state.squeeze().numpy())
        ax[1].cla()
        ax[1].imshow(next_state.squeeze().numpy())
        ax[1].set_title(f'action: {actions_dict[rand_action]}')
        counter = counter + curr_state
        ax[2].cla()
        ax[2].imshow(counter.squeeze().numpy())
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