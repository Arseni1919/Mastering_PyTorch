import torch
import gymnasium as gym
import gymnasium_robotics

from define_model import PointMazeTranslationPredictor

gym.register_envs(gymnasium_robotics)

env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100, render_mode='human', continuing_task=False)
# env = gym.make('PointMaze_Large-v3', max_episode_steps=100, render_mode='human')
# env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100)

predictor = PointMazeTranslationPredictor(out_features=4)
predictor.load_state_dict(torch.load('point_maze_predictor.pt'))
predictor.eval()

curr_obs, info = env.reset()
# curr_obs, info = env.reset(seed=42)
epoch = 0
while epoch < 11:
    actions_list = []
    curr_state = torch.tensor(curr_obs['observation'])
    for j in range(100):
        action = env.action_space.sample()
        # goal_state = torch.cat([torch.tensor(curr_obs['desired_goal']), torch.zeros(2)])
        goal_state = torch.tensor(curr_obs['desired_goal'])
        next_state = predictor(curr_state, torch.tensor(action))
        # distance = (curr_state[:2] - goal_state).norm(dim=-1).item()
        distance = torch.linalg.vector_norm(next_state[:2] - goal_state).item()
        actions_list.append([distance, action])
    min_action = min(actions_list, key=lambda x: x[0])[1]
    next_obs, reward, terminated, truncated, info = env.step(min_action)

    if terminated or truncated:
        epoch += 1
        curr_obs, info = env.reset()
        continue

    curr_obs = next_obs

env.close()
