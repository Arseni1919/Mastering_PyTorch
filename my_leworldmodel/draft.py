import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100, render_mode='rgb_array', width=224, height=224)
# env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100, render_mode='human')
# env = gym.make('PointMaze_Large-v3', max_episode_steps=100, render_mode='human')
# env = gym.make('PointMaze_UMaze-v3', max_episode_steps=100)
pe = env.unwrapped.point_env
pe.mujoco_renderer.default_cam_config = dict(
    distance=5.5, elevation=-90.0, azimuth=90.0, lookat=np.zeros(3))
pe.mujoco_renderer.close()          # forces the viewer to rebuild with the new config
m = pe.model
for i in range(m.nsite):
    if 'target' in m.site(i).name:
        m.site_rgba[i, 3] = 0.0

observation, info = env.reset(seed=42)
curr_img = env.render()
for i in range(1000):
    print(f'\rstep: {i}', end='')
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    next_img = env.render()

    if terminated or truncated:
        observation, info = env.reset()
    plt.cla()
    plt.imshow(next_img)
    plt.pause(0.001)
env.close()
