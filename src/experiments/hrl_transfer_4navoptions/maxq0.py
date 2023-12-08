import argparse
import copy
import os
import random
from custom.taxi.translation import *
from custom.taxi.taxi_v4 import Taxi2PEnv
import gym
from tqdm import tqdm
import numpy as np
from src.utils.utils import save_training

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EXPERIMENT = "hrl_transfer4nav"
EXPERIMENT_FOLDER = os.path.join(PROJECT_DIR, "res", EXPERIMENT)

'''
The idea is that we are going to copy the tables for FETCH_CLIENT1, DRIVE_CLIENT1 to FETCH_CLIENT2, DRIVE_CLIENT2
The new root state is the only state with the whole view of the state space. Once the root state selects to either 
FETCH_CLIENT1, FETCH_CLIENT2, DRIVE_CLIENT1 or DRIVE_CLIENT2 the state space is reduced to the respective client. This
way, we can extrapolate the knowledge learned with one client to two clients.

'''
MAPPINGS = {
    10: 12,
    11: 13
}


class HRLAgent2P:

    def __init__(self, env, alpha, gamma, epsilon, render=False, vtable=None, ctable=None):
        self.fitted = False
        self.done = False
        self.frozen = False
        self.render = False
        self.reward = 0
        self.state_prime = None
        self.state_prime_lowd = None
        self.is_state_translated = False
        self.RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]

        self.passenger = None  # This will control the passenger for which we are translating the state for.
        # For the ROOT Task, the passenger is None, as we have the whole view of the state space.

        self.env = env
        self.SOUTH = 0
        self.NORTH = 1
        self.EAST = 2
        self.WEST = 3
        self.PICKUP = 4
        self.DROPOFF = 5

        self.NAVIGATE_R = 6  # Navigate to source
        self.NAVIGATE_G = 7  # Navigate to destination
        self.NAVIGATE_B = 8  # Navigate to destination
        self.NAVIGATE_Y = 9  # Navigate to destination

        self.FETCH_CLIENT1 = 10
        self.DRIVE_CLIENT1 = 11
        self.FETCH_CLIENT2 = 12
        self.DRIVE_CLIENT2 = 13

        self.ROOT = 14
        self.actions = [

            # Acciones primitivas

            [self.SOUTH],  # 0
            [self.NORTH],  # 1
            [self.EAST],  # 2
            [self.WEST],  # 3
            [self.PICKUP],  # 4
            [self.DROPOFF],  # 5

            # Opciones
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 6: NAVIGATEr
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 7: NAVIGATEg
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 8: NAVIGATEb
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 9: NAVIGATEy
            [self.NAVIGATE_R, self.NAVIGATE_G, self.NAVIGATE_B, self.NAVIGATE_Y, self.PICKUP],  # 10: FETCH_CLIENT
            [self.NAVIGATE_R, self.NAVIGATE_G, self.NAVIGATE_B, self.NAVIGATE_Y, self.DROPOFF],  # 11: FETCH_CLIENT
            [self.NAVIGATE_R, self.NAVIGATE_G, self.NAVIGATE_B, self.NAVIGATE_Y, self.PICKUP],  # 12: FETCH_CLIENT
            [self.NAVIGATE_R, self.NAVIGATE_G, self.NAVIGATE_B, self.NAVIGATE_Y, self.DROPOFF],  # 13: FETCH_CLIENT

            [self.FETCH_CLIENT1, self.DRIVE_CLIENT1, self.FETCH_CLIENT2, self.DRIVE_CLIENT2]  # 12: ROOT
        ]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        assert vtable is None and ctable is None or vtable is not None and ctable is not None, "Both tables must passed or none of them"
        if not vtable is None and not ctable is None:
            self.load_tables(vtable, ctable)
            if self.vtable.shape != (len(self.actions), self.env.observation_space.n):
                self.extend_tables()
            else:
                self.fitted = True

        else:
            self.vtable = np.zeros((len(self.actions), self.env.observation_space.n), dtype='float64')
            self.C = np.zeros((len(self.actions), self.env.observation_space.n, len(self.actions)), dtype='float64')

    def extend_tables(self):
        vtable = np.zeros((len(self.actions), self.env.observation_space.n), dtype='float64')
        ctable = np.zeros((len(self.actions), self.env.observation_space.n, len(self.actions)), dtype='float64')
        vtable[:self.vtable.shape[0], :self.vtable.shape[1]] = self.vtable
        ctable[:self.C.shape[0], :self.C.shape[1], :self.C.shape[2]] = self.C
        self.vtable = vtable
        self.C = ctable

        # Learned knowledge is transfered to the new actions, as they are the same but different people,
        # then, we will finetune ROOT to select the correct action at each moment
        for old_action, new_action in MAPPINGS.items():
            self.vtable[new_action] = self.vtable[old_action]
            self.C[new_action] = self.C[old_action]

    def train(self):
        self.frozen = False
        self.done = False
        self.reward = 0

    def eval(self):
        self.frozen = True
        self.done = False
        self.render = True
        self.reward = 0

    def is_primitive(self, act):
        return act < 6

    def is_terminal(self, a, done):

        taxi_row, taxi_col, pass_loc1, pass_loc2, dest_idx1, dest_idx2 = list(self.env.decode(self.env.s))
        taxiloc = (taxi_row, taxi_col)

        if a == self.ROOT:
            result = done
        elif a == self.NAVIGATE_R:
            result = taxiloc == self.RGBY[0]
        elif a == self.NAVIGATE_G:
            result = taxiloc == self.RGBY[1]
        elif a == self.NAVIGATE_B:

            result = taxiloc == self.RGBY[2]
        elif a == self.NAVIGATE_Y:
            result = taxiloc == self.RGBY[3]
        elif a == self.DRIVE_CLIENT1:
            result = pass_loc1 != 4
        elif a == self.FETCH_CLIENT1:
            result = pass_loc1 == 4
        elif a == self.DRIVE_CLIENT2:
            result = pass_loc2 != 4
        elif a == self.FETCH_CLIENT2:
            result = pass_loc2 == 4
        else:
            result = self.is_primitive(a)
        return 1 if result else 0

    def greedy_policy(self, action, state):
        possible_actions = [action_prime for action_prime in self.actions[action]
                            if self.is_primitive(action_prime) or not self.is_terminal(action_prime, self.done)]

        q = np.array([self.Q(action, state, action_prime) for action_prime in
                      possible_actions])
        max_arg = np.argmax(q)
        return possible_actions[max_arg] if np.random.rand(1) >= self.epsilon else random.choice(possible_actions)

    def argmaxq(self, action, state):
        possible_actions = [action_prime for action_prime in self.actions[action]
                            if self.is_primitive(action_prime) or not self.is_terminal(action_prime, self.done)]

        q = np.array([self.Q(action, state, action_prime) for action_prime in possible_actions])
        max_arg = np.argmax(q)
        return possible_actions[max_arg]

    def Q(self, action, state, action_prime):
        # if Task is ROOT, we need to translate the state to the correct passenger for the action_prime, as ROOT
        # is the only one that has the whole view of the state space
        if action == self.ROOT:
            translated_state = translate(state, 1 if action_prime in [self.FETCH_CLIENT1, self.DRIVE_CLIENT1] else 2)
            return self.V(action_prime, translated_state) + self.C[action, state, action_prime]
        return self.V(action_prime, state) + self.C[action, state, action_prime]

    def V(self, action, state):
        if self.is_primitive(action):
            return self.vtable[action, state]
        else:
            return self.Q(action, state, self.argmaxq(action, state))

    def is_taxi_at(self, taxiloc, destidx):
        return taxiloc == self.RGBY[destidx]

    def is_passenger_in_taxi(self, passidx):
        return passidx == 4

    def decode(self, state):
        return list(self.env.decode(state))

    def max_q(self, action, state):
        if self.done:
            action = self.ROOT + 1

        if action in [self.FETCH_CLIENT1, self.DRIVE_CLIENT1]:
            self.passenger = 1
        elif action in [self.FETCH_CLIENT2, self.DRIVE_CLIENT2]:
            self.passenger = 2

        if action < self.ROOT and not self.is_state_translated:
            state = translate(state, self.passenger)
            self.is_state_translated = True

        if self.is_primitive(action):
            self.state_prime, reward, self.done, truncated, info = self.env.step(action)
            assert self.passenger is not None, "Passenger must be set"
            self.state_prime_lowd = translate(self.state_prime, self.passenger)

            if self.render:
                self.env.render()
            if self.fitted:
                print(f"\rScore: {reward}", end="")
            self.done = self.done or truncated
            self.reward += reward
            if not self.frozen:
                self.vtable[action, state] += self.alpha * (float(reward) - self.vtable[action, state])
            return 1
        elif action <= self.ROOT:
            count = 0
            while not self.done and not self.is_terminal(action, self.done):
                action_prime = self.greedy_policy(action, state)
                if action_prime == self.PICKUP:
                    print("pickup")

                # Note that after this line, state and state_prime, change.
                # Therefore, we need to translate back for root
                N = self.max_q(action_prime, state)

                if action == self.ROOT:
                    state_prime = self.state_prime
                    self.is_state_translated = False
                else:
                    state_prime = self.state_prime_lowd

                self.frozen = action < self.ROOT
                if not self.frozen:
                    self.C[action, state, action_prime] += self.alpha * (
                            (self.gamma ** N) * self.V(action, state_prime) - self.C[action, state, action_prime])
                count += N
                state = state_prime
            return count

    def run(self, episodes, steps, render_training=False, show_plot=False):
        rewards = []

        for j in tqdm(range(episodes), colour="green", desc="episode", position=0):
            # if j == 5000:
            #     print('fitted')
            state, info = self.env.reset()
            self.train()
            count = self.max_q(self.ROOT, state)
            rewards.append(agent.reward)
        else:
            self.fitted = True
            save_training(
                experiment_name=EXPERIMENT,
                config={
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "episodes": episodes,
                    "steps": steps,
                    "render_training": render_training,
                    "show_plot": show_plot
                },
                table={
                    "vtable": self.vtable,
                    "ctable": self.C
                },
                rewards=rewards, n=100,
                show_plot=show_plot
            )

    def test(self):
        assert self.fitted, "Agent not fitted"
        self.eval()
        state, info = self.env.reset()
        rewards = 0
        count = self.max_q(self.ROOT, state)
        rewards += agent.reward

        self.env.close()

    def load_tables(self, vtable, ctable):
        self.vtable = np.load(vtable)
        self.C = np.load(ctable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--render_training", action="store_true")
    args = parser.parse_args()
    do_train = not args.experiment.startswith(EXPERIMENT) or args.experiment is None

    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    render_training = args.render_training
    episodes = args.episodes
    steps = args.steps
    show_plot = args.show_plot

    config = {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": args.epsilon,
        "episodes": episodes,
        "steps": steps,
        "render_training": render_training,
        "show_plot": show_plot
    }

    env = gym.make("custom/Taxi-v1.7", render_mode="human" if render_training or not do_train else None)
    state_size = env.observation_space.n
    path = os.path.join(PROJECT_DIR, "res", args.experiment)
    agent = HRLAgent2P(env,
                       alpha,
                       gamma,
                       epsilon,
                       vtable=os.path.join(path, "vtable.npy"),
                       ctable=os.path.join(path, "ctable.npy")
                       )

    if do_train:
        agent.run(episodes, steps, render_training, show_plot)
        env.close()
        agent.env = gym.make("custom/Taxi-v1.7", render_mode="human")

    agent.test()
