import matplotlib.pyplot as plt
import torch

from define_model import Encoder, TranslationPredictor, Predictor
from get_data import create_curr_state, create_next_state

N_ACTIONS = 5
ACTION_NAMES = {0: 'stay', 1: 'forward', 2: 'backward', 3: 'right', 4: 'left'}


def plot_field(encoder, side):
    all_images = torch.eye(side * side).reshape(side * side, 1, side, side)
    encodings = encoder(all_images).detach().numpy()
    x = [i[0] for i in encodings]
    y = [i[1] for i in encodings]
    plt.scatter(x, y)
    plt.show()


def agent_pos(state: torch.Tensor):
    row, col = (state[0] == 1).nonzero()[0].tolist()
    return row, col


def plan_action(predictor, z_curr: torch.Tensor, z_goal: torch.Tensor):
    """Imagine all 5 actions with the world model, keep the one landing closest to the goal.

    Only the predictor and the two embeddings are used here -- the true dynamics are never
    queried, so this actually tests the world model rather than the encoder alone.
    """
    z_next = predictor(z_curr.expand(N_ACTIONS, -1), torch.arange(N_ACTIONS))
    distances = (z_next - z_goal).norm(dim=-1)
    action = distances.argmin().item()
    return action, distances[action].item()


def navigate(encoder, predictor, start: torch.Tensor, goal: torch.Tensor, budget: int = 40):
    """Closed loop: encode the real state -> plan one step -> execute -> re-encode.

    Re-encoding every step means predictor error never compounds; the world model only has
    to be right one step at a time.
    """
    state = start
    z_goal = encoder(goal.unsqueeze(0))
    trajectory = [(state, None, None)]

    for _ in range(budget):
        if torch.equal(state, goal):
            break
        action, distance = plan_action(predictor, encoder(state.unsqueeze(0)), z_goal)
        state = create_next_state(state, action)
        trajectory.append((state, action, distance))

    return trajectory, torch.equal(state, goal)



def plan_action_latent(predictor, z_curr: torch.Tensor, z_goal: torch.Tensor):
    z_next = predictor(z_curr.expand(N_ACTIONS, -1), torch.arange(N_ACTIONS))
    distances = (z_next - z_goal).norm(dim=-1)
    action = distances.argmin().item()
    return action, distances[action].item(), z_next[action]


def navigate_latent(encoder, predictor, start: torch.Tensor, goal: torch.Tensor, budget: int = 40):
    z_curr = encoder(start.unsqueeze(0))
    curr_state = start
    z_goal = encoder(goal.unsqueeze(0))
    trajectory = []
    states = [curr_state]
    for _ in range(budget):
        distance = (z_curr - z_goal).norm(dim=-1).item()
        if distance < 0.1:
            break
        action, distance, z_next = plan_action_latent(predictor, z_curr, z_goal)
        next_state = create_next_state(curr_state, action)
        trajectory.append((z_next, action, distance))
        states.append(next_state)
        curr_state = next_state
        z_curr = z_next
    return trajectory, torch.equal(goal, states[-1])


def evaluate_latent(encoder, predictor, n: int = 200, side: int = 15, budget: int = 40, seed: int = 1):
    torch.manual_seed(seed)
    hits = 0
    with torch.no_grad():
        for _ in range(n):
            start, goal = create_curr_state(side), create_curr_state(side)
            trajectory, reached = navigate_latent(encoder, predictor, start, goal, budget)
            hits += reached
    return hits / n



def animate_latent(encoder, predictor, side: int = 15, budget: int = 20, n_episodes: int = 10):
    with torch.no_grad():
        for _ in range(n_episodes):
            start, goal = create_curr_state(side), create_curr_state(side)
            trajectory, reached = navigate_latent(encoder, predictor, start, goal, budget)
            actions = [i[1] for i in trajectory]
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[1].imshow(goal.squeeze().numpy())
            ax[1].set_title(f'goal {agent_pos(goal)}')
            curr_state = start
            for i, action in enumerate(actions):
                ax[0].cla()
                next_state = create_next_state(curr_state, action)
                ax[0].imshow(next_state.squeeze().numpy())
                label = 'start' if action is None else f'{ACTION_NAMES[action]}'
                ax[0].set_title(f'step {i}: {label}')
                plt.pause(0.15)

                curr_state = next_state

            fig.suptitle('reached the goal' if reached else 'did NOT reach the goal')
            plt.show()


def evaluate(encoder, predictor, n: int = 200, side: int = 15, budget: int = 40, seed: int = 1):
    """Success rate over n random (start, goal) pairs."""
    torch.manual_seed(seed)
    hits = 0
    final_distances = []

    with torch.no_grad():
        for _ in range(n):
            start, goal = create_curr_state(side), create_curr_state(side)
            trajectory, reached = navigate(encoder, predictor, start, goal, budget)
            hits += reached
            (sr, sc), (gr, gc) = agent_pos(trajectory[-1][0]), agent_pos(goal)
            final_distances.append(abs(sr - gr) + abs(sc - gc))

    return hits / n, sum(final_distances) / len(final_distances)


def animate(encoder, predictor, side: int = 15, budget: int = 40, n_episodes: int = 10):
    with torch.no_grad():
        for _ in range(n_episodes):
            start, goal = create_curr_state(side), create_curr_state(side)
            trajectory, reached = navigate(encoder, predictor, start, goal, budget)

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[1].imshow(goal.squeeze().numpy())
            ax[1].set_title(f'goal {agent_pos(goal)}')
            for i, (state, action, distance) in enumerate(trajectory):
                ax[0].cla()
                ax[0].imshow(state.squeeze().numpy())
                label = 'start' if action is None else f'{ACTION_NAMES[action]} | d={distance:.2f}'
                ax[0].set_title(f'step {i}: {label}')
                plt.pause(0.25)
            fig.suptitle('reached the goal' if reached else 'did NOT reach the goal')
            plt.show()


def main():
    side = 15
    out_features = 2

    encoder = Encoder(in_features=side**2, out_features=out_features)
    encoder.load_state_dict(torch.load('encoder.pt'))
    predictor = TranslationPredictor(out_features=out_features)
    # predictor = Predictor(out_features=out_features)
    predictor.load_state_dict(torch.load('predictor.pt'))
    encoder.eval()
    predictor.eval()

    # success, final_distance = evaluate(encoder, predictor, side=side)
    # print(f'success rate: {success:.1%} | mean final manhattan distance: {final_distance:.2f}')
    #
    # animate(encoder, predictor, side=side)

    success = evaluate_latent(encoder, predictor, side=side)
    print(f'success rate: {success:.1%}')

    animate_latent(encoder, predictor, side=side)

    plot_field(encoder, side)




if __name__ == '__main__':
    main()
