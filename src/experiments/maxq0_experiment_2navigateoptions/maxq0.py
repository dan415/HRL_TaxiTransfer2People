import argparse
import copy
import os
import random

from tqdm import tqdm
import gymnasium as gym
import numpy as np
from src.utils.utils import save_training

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EXPERIMENT = "maxq02nav"
EXPERIMENT_FOLDER = os.path.join(PROJECT_DIR, "res", EXPERIMENT)


class HRLAgent:

    def __init__(self, env, alpha, gamma, epsilon, render=False):
        self.fitted = False
        self.done = False
        self.frozen = False
        self.render = False
        self.reward = 0
        self.state_prime = None
        self.RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]

        self.env = env
        self.SOUTH = 0
        self.NORTH = 1
        self.EAST = 2
        self.WEST = 3
        self.PICKUP = 4
        self.DROPOFF = 5

        self.NAVIGATE_S = 6 # Navigate to source
        self.NAVIGATE_D = 7 # Navigate to destination

        self.FETCH_CLIENT = 8
        self.DRIVE_CLIENT = 9


        self.ROOT = 10
        self.actions = [

            # Acciones primitivas

            [self.SOUTH],  # 0
            [self.NORTH],  # 1
            [self.EAST],  # 2
            [self.WEST],  # 3
            [self.PICKUP],  # 4
            [self.DROPOFF],  # 5

            # Opciones
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 6: NAVIGATES
            [self.SOUTH, self.NORTH, self.EAST, self.WEST],  # 7: NAVIGATED
            [self.NAVIGATE_S, self.PICKUP],  # 8: FETCH_CLIENT
            [self.NAVIGATE_D, self.DROPOFF],  # 9: DRIVE_CLIENT

            [self.FETCH_CLIENT, self.DRIVE_CLIENT]  # 10: ROOT
        ]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.vtable = np.zeros((len(self.actions), self.env.observation_space.n), dtype='float64')
        self.C = np.zeros((len(self.actions), self.env.observation_space.n, len(self.actions)), dtype='float64')

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

        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
        taxiloc = (taxirow, taxicol)

        if a == self.ROOT:
            result = done
        elif a == self.NAVIGATE_S:
            result = passidx != 4 and taxiloc == self.RGBY[passidx]
        elif a == self.NAVIGATE_D:
            result = passidx == 4 and taxiloc == self.RGBY[destidx]
        elif a == self.DRIVE_CLIENT:
            result = passidx != 4
        elif a == self.FETCH_CLIENT:
            result = passidx == 4
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
        if self.is_primitive(action):
            self.state_prime, reward, self.done, truncated, info = self.env.step(action)

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
                N = self.max_q(action_prime, state)
                if not self.frozen:
                    self.C[action, state, action_prime] += self.alpha * (
                            (self.gamma ** N) * self.V(action, self.state_prime) - self.C[action, state, action_prime])
                count += N
                state = self.state_prime
            return count

    def run(self, episodes, steps, render_training=False, show_plot=False):
        rewards = []
        rewards_test = []
        for j in tqdm(range(episodes), colour="green", desc="episode", position=0):
            # if j == 5000:
            #     print('fitted')
            state, info = self.env.reset()
            self.train()
            count = self.max_q(self.ROOT, state)
            rewards.append(agent.reward)

            self.eval()
            state, info = self.env.reset()
            _ = self.max_q(self.ROOT, state)
            rewards_test.append(agent.reward)
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
                test_rewards=rewards_test,
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
        self.vtable = vtable
        self.C = ctable
        self.fitted = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtable", type=str, default=None)
    parser.add_argument("--ctable", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--render_training", action="store_true")
    args = parser.parse_args()
    do_train = args.vtable is None or args.ctable is None

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

    env = gym.make("Taxi-v3", render_mode="human" if render_training or not do_train else None)
    state_size = env.observation_space.n

    agent = HRLAgent(env, alpha, gamma, epsilon)

    if do_train:
        agent.run(episodes, steps, render_training, show_plot)
        env.close()
        agent.env = gym.make("Taxi-v3", render_mode="human")
    else:
        agent.load_tables(np.load(os.path.join(EXPERIMENT_FOLDER, args.vtable)),
                          np.load(os.path.join(EXPERIMENT_FOLDER, args.ctable)))

    agent.test()
