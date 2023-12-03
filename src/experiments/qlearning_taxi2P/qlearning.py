import argparse
import os
import random

from tqdm import tqdm
import gymnasium as gym
import numpy as np
from src.utils.utils import save_training
from src.environment.taxi_v4 import *

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EXPERIMENT = "qlearning_taxi2P"
EXPERIMENT_FOLDER = os.path.join(PROJECT_DIR, "res", EXPERIMENT)


def learning_step(env, state, epsilon):
    if render_training:
        env.render()
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qtable[state, :])

    new_state, reward, terminated, truncated, info = env.step(action)

    qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

    return new_state, terminated, truncated, reward


def train(show_plot=False):
    rewards = []
    epsilon = 1.0
    reward = 0
    for episode in tqdm(range(episodes), colour="green", desc="episode", position=0):
        state, info = env.reset()

        for _ in range(steps):
            state, terminated, truncated, reward = learning_step(env, state, epsilon)

            done = truncated or terminated
            if done:
                break
        epsilon = np.exp(-decay_rate * episode)
        rewards.append(reward)
    else:
        save_training(experiment_name=EXPERIMENT, config=config, table=qtable, rewards=rewards, n=100,
                      show_plot=show_plot)


def test():
    state, info = env.reset()
    rewards = 0

    for s in range(steps):

        action = np.argmax(qtable[state, :])
        state, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        rewards += reward
        env.render()
        print(f"\rScore: {rewards}", end="")
        if done:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.9)
    parser.add_argument("--discount_rate", type=float, default=0.8)
    parser.add_argument("--decay_rate", type=float, default=0.005)
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()
    do_train = True

    learning_rate = args.learning_rate
    discount_rate = args.discount_rate
    decay_rate = args.decay_rate
    render_training = False
    episodes = args.episodes
    steps = args.steps
    show_plot = args.show_plot

    config = {
        "learning_rate": learning_rate,
        "discount_rate": discount_rate,
        "decay_rate": decay_rate,
        "episodes": episodes,
        "steps": steps,
        "render_training": render_training,
        "show_plot": show_plot
    }

    if args.qtable:
        qtable = np.load(os.path.join(EXPERIMENT_FOLDER, args.qtable))
        do_train = False

    env = gym.make("Taxi2p-v1", render_mode="human" if render_training or not do_train else None)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    if do_train:
        train(show_plot=show_plot)
        env.close()
        env = gym.make("Taxi2p-v1", render_mode="human")

    test()
    env.close()
