import matplotlib.pyplot as plt

from define_model import Encoder, Predictor
from get_data import get_nav_data, create_curr_state, create_next_state

import torch
import torch.nn.functional as F


def get_closest_next_state(encoder, curr_state, goal_state):
    distance_list = []
    for action in range(5):
        next_state = create_next_state(curr_state, action)
        next_state_emb = encoder(next_state.unsqueeze(0))
        goal_state_emb = encoder(goal_state.unsqueeze(0))
        # distance = F.cosine_similarity(next_state_emb, goal_state_emb, dim=-1)
        distance = (next_state_emb - goal_state_emb).norm(p=2)
        distance_list.append((action, next_state, distance))
    action, next_state, distance = max(distance_list, key=lambda t: t[2])
    return action, next_state, distance


def main():
    N = 100
    side = 15
    out_features = 16

    encoder = Encoder(in_features=side**2, out_features=out_features)
    encoder.load_state_dict(torch.load('encoder.pt'))
    predictor = Predictor(out_features=out_features)
    predictor.load_state_dict(torch.load('predictor.pt'))

    for i_try in range(10):
        curr_state = create_curr_state(side=side)
        plt.imshow(curr_state.squeeze().detach().numpy())
        plt.show()
        goal_state = create_curr_state(side=side)
        plt.imshow(goal_state.squeeze().detach().numpy())
        plt.show()

        visited_states = [curr_state]
        for i in range(20):
            action, next_state, distance = get_closest_next_state(encoder, curr_state, goal_state)
            visited_states.append(next_state)

            plt.cla()
            plt.imshow(curr_state.squeeze().detach().numpy())
            plt.title(f'iter: {i} | action: {action} | distance: {distance.item()}')
            plt.pause(0.1)

            curr_state = next_state

        plt.show()

    # data = get_nav_data(N=N, side=side)
    # for curr_state, rand_action, next_state in data:
    #     input_item = curr_state.unsqueeze(0)
    #     emb = encoder(input_item)
    #
    #     emb = emb.squeeze().reshape(4, 4).detach().numpy()
    #     plt.imshow(emb)
    #     plt.show()




if __name__ == '__main__':
    main()
