"""Goal-reaching in PointMaze from IMAGES ONLY, with a JEPA world model.

Nothing privileged is used anywhere: no ground-truth state, no maze layout. The inputs are
rendered frames, and the goal is a rendered frame.

The planner is CEM over the learned predictor (INSTRUCTIONS section 8), but steered towards
waypoints taken from a graph of encoded observations. That structure is what makes it work:

  The latent metric is unreliable at long range -- corr(latent distance, true distance) is
  only 0.39, because the JEPA objective leaves the manifold curved (the failure the
  Temporal Straightening paper exists to fix). But a curved manifold is still locally
  Euclidean, so short-range comparisons are fine. The graph therefore carries the long-range
  structure, and CEM is only ever asked to judge a five-block rollout against a nearby
  waypoint.

Two details that each cost a measured chunk of success rate:
  - Waypoints are REAL encoded observations, so the cost compares like with like. Comparing
    against the goal image directly asks the planner to arrive AND stop, because a goal
    rendered at rest carries zero velocity while the latent encodes velocity (40% -> 85%).
  - The waypoint is chosen by DISTANCE along the path, not by node count. Node count
    silently collapses onto the goal, since each hop is only one transition long.
"""
import collections
import heapq

import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import mujoco
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from torchvision import transforms

from define_model import ConvEncoder, LatentPredictor, LinearPredictor
from get_data import get_point_maze_rgb_env, PointMazeRGBDataset

gym.register_envs(gymnasium_robotics)

VIEW = True             # show the frames the encoder sees
EPISODES = 40
HISTORY = 3
FRAMESKIP = 5
LATENT_DIM = 8
PREDICTOR_TYPE = 'mlp'  # must match what train_point_maze_rgb.py was run with

# --- CEM (INSTRUCTIONS section 8). One predictor step == one FRAMESKIP block.
PLAN_HORIZON = 5        # H: how many blocks ahead to plan
CEM_SAMPLES = 300       # candidate action SEQUENCES per iteration
CEM_ITERS = 10          # refit rounds
CEM_ELITES = 30         # best sequences kept to refit the distribution
EXECUTE_K = 1           # blocks executed before replanning (receding horizon)

# --- waypoint graph over encoded observations
GRAPH_N = 6000          # transitions used to build it
WAYPOINT_AHEAD = 10     # waypoint this many transition-lengths along the path
GOAL_DIRECT = 2         # within this many transition-lengths, steer at the goal itself


def make_obs(transform, frames):
    """the (3*HISTORY, 64, 64) observation from the rolling frame buffer"""
    return torch.cat([transform(f) for f in frames], dim=0).unsqueeze(0)


def render_goal_obs(env, transform, goal_xy):
    """Render the goal, then put the simulator back exactly as it was.

    Done on the main env on purpose: a second env means a second MuJoCo GL context, and
    interleaving renders across contexts glitches the display.
    """
    pe = env.unwrapped.point_env
    qpos, qvel = pe.data.qpos.copy(), pe.data.qvel.copy()
    pe.data.qpos[:2] = goal_xy
    pe.data.qvel[:2] = 0.0
    mujoco.mj_forward(pe.model, pe.data)
    raw = env.render().copy()
    frame = transform(raw)
    pe.data.qpos[:] = qpos
    pe.data.qvel[:] = qvel
    mujoco.mj_forward(pe.model, pe.data)
    return torch.cat([frame] * HISTORY, dim=0).unsqueeze(0), raw


def build_latent_graph(encoder, transform):
    """Nodes are ENCODED observations; edges are transitions that really happened.

    Temporal edges join consecutive observations within an episode, so no edge can cross a
    wall. Proximity edges (radius = the median transition length, taken from the data rather
    than tuned) join the per-episode chains into one connected graph.
    """
    dataset = PointMazeRGBDataset(N=GRAPH_N)
    with torch.no_grad():
        latents = torch.cat([encoder(batch[0]) for batch in DataLoader(dataset, batch_size=256)])
    points = latents.numpy()
    frame_index = [i for i, _ in dataset.transitions]

    adjacency = [[] for _ in points]
    lengths = []
    for k in range(len(frame_index) - 1):
        if frame_index[k + 1] == frame_index[k] + 1:
            w = float(np.linalg.norm(points[k] - points[k + 1]))
            adjacency[k].append((k + 1, w))
            adjacency[k + 1].append((k, w))
            lengths.append(w)
    step_length = float(np.median(lengths))

    for ia, ib in cKDTree(points).query_pairs(step_length):
        w = float(np.linalg.norm(points[ia] - points[ib]))
        adjacency[ia].append((ib, w))
        adjacency[ib].append((ia, w))

    reached = {0}
    queue = collections.deque([0])
    while queue:
        u = queue.popleft()
        for v, _ in adjacency[u]:
            if v not in reached:
                reached.add(v)
                queue.append(v)
    print(f'latent graph: {len(points)} nodes | median transition length {step_length:.3f} '
          f'| largest component {len(reached) / len(points):.1%}')
    # ground-truth positions of each node, used ONLY to draw the debug plot below --
    # the planner never sees them
    true_xy = np.array([dataset.get_states(k)[0][:2].numpy() for k in range(len(points))])
    return latents, points, adjacency, step_length, true_xy


def plot_latent_graph(points, adjacency, true_xy, max_edges=4000):
    """Draw the learned graph before planning starts.

    Left: the graph as the PLANNER sees it -- nodes projected to 2-D by PCA of the latents,
    which is all the method has. Right: the same nodes and edges at their true maze
    positions. The right panel uses ground truth for DISPLAY ONLY; it is the quick check
    that no edge jumps a wall, which is the property the whole approach rests on.
    """
    centred = points - points.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[:2].T

    edges = [(u, v) for u, nbrs in enumerate(adjacency) for v, _ in nbrs if u < v]
    if len(edges) > max_edges:
        edges = [edges[i] for i in np.random.choice(len(edges), max_edges, replace=False)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for a, xy, title in [(ax[0], projected, 'graph in LATENT space (PCA of the latents)'),
                         (ax[1], true_xy, 'same graph at true maze positions (display only)')]:
        segments = np.array([[xy[u], xy[v]] for u, v in edges])
        a.plot(segments[:, :, 0].T, segments[:, :, 1].T, color='steelblue', lw=0.2, alpha=0.3)
        a.scatter(xy[:, 0], xy[:, 1], s=3, color='crimson', zorder=3)
        a.set_title(title)
        a.set_aspect('equal')
    fig.suptitle(f'waypoint graph: {len(points)} nodes, {len(edges)} of the edges drawn')
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(2.0)


def shortest_path(adjacency, source, target):
    dist = {source: 0.0}
    prev = {}
    queue = [(0.0, source)]
    visited = set()
    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(queue, (nd, v))
    if target not in visited:
        return None
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return path[::-1]


def pick_waypoint(latents, points, adjacency, step_length, z_curr, z_goal):
    """A nearby encoded observation on the geodesic path towards the goal."""
    here, goal = z_curr[0].numpy(), z_goal[0].numpy()
    if np.linalg.norm(here - goal) < GOAL_DIRECT * step_length:
        return z_goal

    source = int(np.argmin(((points - here) ** 2).sum(1)))
    target = int(np.argmin(((points - goal) ** 2).sum(1)))
    path = shortest_path(adjacency, source, target)
    if path is None:
        return z_goal

    travelled, idx = 0.0, 0
    for j in range(1, len(path)):
        travelled += float(np.linalg.norm(points[path[j]] - points[path[j - 1]]))
        idx = j
        if travelled >= WAYPOINT_AHEAD * step_length:
            break
    return latents[path[idx]].unsqueeze(0)


def cem_plan(predictor, z_curr, z_target):
    """Cross-Entropy Method: sample action SEQUENCES, roll them through the predictor, keep
    the elites whose rollout passes closest to the target, refit, repeat.

    The cost is averaged over the rollout weighted towards the later steps rather than taken
    at the terminal state alone: a terminal-only cost is flat across candidates whenever the
    target is further than H blocks away, leaving CEM nothing to optimise.
    """
    mu = torch.zeros(PLAN_HORIZON, 2)
    sigma = torch.ones(PLAN_HORIZON, 2)
    weights = torch.linspace(0.5, 1.5, PLAN_HORIZON)[:, None]

    for _ in range(CEM_ITERS):
        noise = torch.randn(CEM_SAMPLES, PLAN_HORIZON, 2)
        candidates = (mu + sigma * noise).clamp(-1.0, 1.0)     # the action space is [-1, 1]

        z = z_curr.expand(CEM_SAMPLES, -1)
        step_costs = []
        for k in range(PLAN_HORIZON):
            z = predictor(z, candidates[:, k])
            step_costs.append((z - z_target).norm(dim=-1))
        cost = (torch.stack(step_costs) * weights).mean(0)

        elites = candidates[cost.topk(CEM_ELITES, largest=False).indices]
        mu = elites.mean(dim=0)
        sigma = elites.std(dim=0) + 1e-6                       # never let it collapse to zero

    return mu, cost


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    env = get_point_maze_rgb_env(max_episode_steps=3000)   # no practical step limit when planning

    encoder = ConvEncoder(in_channels=3 * HISTORY, latent_dim=LATENT_DIM)
    encoder.load_state_dict(torch.load(f'point_maze_rgb_conv_encoder_{PREDICTOR_TYPE}.pt'))
    encoder.eval()
    if PREDICTOR_TYPE == 'linear':
        predictor = LinearPredictor(latent_dim=LATENT_DIM, action_dim=2)
    else:
        predictor = LatentPredictor(latent_dim=LATENT_DIM, action_dim=2)
    predictor.load_state_dict(torch.load(f'point_maze_rgb_{PREDICTOR_TYPE}_predictor.pt'))
    predictor.eval()

    latents, points, adjacency, step_length, true_xy = build_latent_graph(encoder, transform)

    view_ax = None
    if VIEW:
        plt.ion()
        plot_latent_graph(points, adjacency, true_xy)
        _, view_ax = plt.subplots(1, 2, figsize=(7, 3.6))

    successes = 0
    for episode in range(EPISODES):
        curr_obs, info = env.reset(seed=1000 + episode)
        frames = [env.render().copy() for _ in range(HISTORY)]
        goal_obs, goal_frame = render_goal_obs(env, transform, curr_obs['desired_goal'])
        with torch.no_grad():
            z_goal = encoder(goal_obs)

        reached = False
        for step in range(200):
            with torch.no_grad():
                z_curr = encoder(make_obs(transform, frames))
                target = pick_waypoint(latents, points, adjacency, step_length, z_curr, z_goal)
                plan, distances = cem_plan(predictor, z_curr, target)

            done = False
            for block in range(EXECUTE_K):                     # receding horizon
                action = plan[block].numpy()
                for _ in range(FRAMESKIP):                     # hold the action, as in training
                    curr_obs, reward, terminated, truncated, info = env.step(action)
                    if reward > 0:
                        reached = True
                    if terminated or truncated:
                        done = True
                        break
                frames = frames[1:] + [env.render().copy()]

                if view_ax is not None:
                    view_ax[0].cla(); view_ax[0].imshow(frames[-1]); view_ax[0].set_axis_off()
                    view_ax[0].set_title(f'ep {episode} step {step}  cost={distances.min():.2f}')
                    view_ax[1].cla(); view_ax[1].imshow(goal_frame); view_ax[1].set_axis_off()
                    view_ax[1].set_title('goal')
                    plt.pause(0.05)

                if reached or done:
                    break
            if reached or done:
                break

        successes += reached
        dist = np.linalg.norm(curr_obs['achieved_goal'] - curr_obs['desired_goal'])
        print(f'episode {episode}: {"reached" if reached else "failed "} | final distance {dist:.3f}')

    print(f'\nsuccess rate: {successes}/{EPISODES} = {successes / EPISODES:.1%}')
    env.close()
    if VIEW:
        plt.ioff()


if __name__ == '__main__':
    main()
