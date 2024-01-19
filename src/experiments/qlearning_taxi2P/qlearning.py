import argparse
import os
import random
from custom.taxi.taxi_v4 import Taxi2PEnv
import gym
from tqdm import tqdm

import numpy as np
from src.utils.utils import save_training

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EXPERIMENT = "qlearning_taxi2P"
EXPERIMENT_FOLDER = os.path.join(PROJECT_DIR, "res", EXPERIMENT)


def learning_step(env, state, epsilon):
    """
    Executes a learning step in the environment with a greedy policy

    :param env: gym environment
    :param state: current state
    :param epsilon: exploration rate

    :return: new_state, terminated, truncated, reward
    """
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
    """
    Trains the agent in the environment, saving the results

    :param show_plot: if True, shows a plot of the rewards

    :return: None
    """
    rewards = []
    rewards_test = []
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

        rewards_test.append(test(render=False))

    else:
        save_training(
            experiment_name=EXPERIMENT,
            config=config,
            table=qtable,
            rewards=rewards,
            n=100,
            show_plot=show_plot,
            test_rewards=rewards_test
        )


def test(render=True):
    """
    Tests the agent in the environment, showing the results and rendering the environment

    :return: rewards
    """
    state, info = env.reset()
    rewards = 0

    for s in range(steps):
        action = np.argmax(qtable[state, :])
        state, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        rewards += reward
        if render:
            env.render()
        print(f"\rScore: {rewards}, Action {action}", end="")
        if done:
            break
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default=None, help="qtable to load, if passed, the agent will not train and will only test")
    parser.add_argument("--episodes", type=int, default=50000, help="number of episodes to train")
    parser.add_argument("--steps", type=int, default=400, help="number of steps per episode")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="learning rate")
    parser.add_argument("--discount_rate", type=float, default=0.8, help="discount rate")
    parser.add_argument("--decay_rate", type=float, default=0.0075, help="decay rate")
    parser.add_argument("--show_plot", action="store_true", help="Whether to show the plot at the end")
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

    # Here we make use of the custom environment Taxi2PEnv
    env = gym.make("custom/Taxi-v1.7", render_mode="human" if render_training or not do_train else None)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    if do_train:
        train(show_plot=show_plot)
        env.close()
        env = gym.make("custom/Taxi-v1.7", render_mode="human")

    test(render=True)
    env.close()
