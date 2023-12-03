import argparse
import copy
import os
import random

from tqdm import tqdm
import gymnasium as gym
import numpy as np
from src.utils.utils import save_training

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
EXPERIMENT = "maxq0"
EXPERIMENT_FOLDER = os.path.join(PROJECT_DIR, "res", EXPERIMENT)

SOUTH = 0
NORTH = 1
EAST = 2
WEST = 3
PICKUP = 4
DROPOFF = 5

FETCH_CLIENT = 6
DRIVE_CLIENT = 7
NAVIGATE = 8
ROOT = 9
LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]
alpha = 0.2
gamma = 1
epsilon = 0.001


def evaluate(action, state):
    possible_actions = [action_prime for action_prime in ACTIONS[action]
                        if is_primitive(action_prime) or not is_terminal_state(action_prime, DONE)]

    q = np.array(
        [Q(action, state, action_prime) + C[action, state, action_prime] for action_prime in possible_actions])
    max_arg = np.argmax(q)
    return possible_actions, max_arg


def greedy_policy(action, state):
    possible_actions, max_arg = evaluate(action, state)
    return possible_actions[max_arg] if random.uniform(0, 1) >= epsilon else random.choice(possible_actions)


def argmaxq(action, state):
    possible_actions, max_arg = evaluate(action, state)
    return possible_actions[max_arg]


def Q(action, state, action_prime):
    return V(action_prime, state) + C[action, state, action_prime]


def V(action, state):
    if is_primitive(action):
        return vtable[action, state]
    else:
        return Q(action, state, argmaxq(action, state))


def is_terminal_state(action, done):
    taxirow, taxicol, passidx, destidx = list(env.decode(env.s))
    taxiloc = (taxirow, taxicol)

    if action == ROOT:
        result = done
    elif action == NAVIGATE:
        result = (is_passenger_in_taxi(passidx) and is_taxi_at(taxiloc, destidx)) or (
                not is_passenger_in_taxi(passidx) and is_taxi_at(taxiloc, passidx))
    elif action == DRIVE_CLIENT:
        result = not is_passenger_in_taxi(passidx)
    elif action == FETCH_CLIENT:
        result = is_passenger_in_taxi(passidx)
    else:
        result = is_primitive(action)
    return 1 if result else 0


def max_q(action, state, frozen=False, render=False):
    global DONE
    if is_primitive(action):
        new_state, reward, DONE, truncated, info = env.step(action)
        if render:
            env.render()
        DONE = DONE or truncated
        if not frozen:
            vtable[action, state] += (1 - alpha) * vtable[action, state] + alpha * float(reward)
        return 1, new_state, reward
    elif action <= ROOT:
        count = 0
        reward = 0
        while not is_terminal_state(action, DONE):
            action_prime = greedy_policy(action, state)
            N, state_prime, r = max_q(action_prime, state)
            if not frozen:
                C[action, state, action_prime] = alpha * (
                            gamma ** N * V(action, state_prime) + (1 - alpha) * C[action, state, action_prime])
            count += N
            state = state_prime
            reward += r
        return count, state, reward


def is_taxi_at(taxiloc, destidx):
    return taxiloc == LOCATIONS[destidx]


def is_passenger_in_taxi(passenger_state):
    return passenger_state == 4


def is_primitive(act):
    return act < 6


def train(show_plot=False):
    rewards = []
    for j in tqdm(range(episodes), colour="green", desc="episode", position=0):
        env.reset()
        reward = 0
        for _ in range(steps):
            count, _, reward = max_q(ROOT, env.s, frozen=False)  # start in root
            if DONE:
                break
        rewards.append(reward)
    else:
        save_training(
            experiment_name=EXPERIMENT,
            config=config,
            table={
                "vtable": vtable,
                "ctable": C
            },
            rewards=rewards, n=100,
            show_plot=show_plot
        )


def test():
    state, info = env.reset()
    rewards = 0
    env.reset()
    for s in range(steps):
        count, _, reward = max_q(ROOT, state, frozen=True, render=True)
        rewards += reward

        print(f"\rScore: {rewards}", end="")
        if DONE:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtable", type=str, default=None)
    parser.add_argument("--ctable", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.9)
    parser.add_argument("--discount_rate", type=float, default=0.8)
    parser.add_argument("--decay_rate", type=float, default=0.005)
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()
    do_train = args.vtable is None or args.ctable is None
    DONE = False

    # ACTIONS = [SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, FETCH_CLIENT, DRIVE_CLIENT, NAVIGATE]
    ACTIONS = [

        # Acciones primitivas

        [SOUTH],  # 0
        [NORTH],  # 1
        [EAST],  # 2
        [WEST],  # 3
        [PICKUP],  # 4
        [DROPOFF],  # 5

        # Opciones
        [NAVIGATE, PICKUP],  # 6: FETCH_CLIENT
        [NAVIGATE, DROPOFF],  # 7: DRIVE_CLIENT
        [SOUTH, NORTH, EAST, WEST],  # 8: NAVIGATE
        [FETCH_CLIENT, DRIVE_CLIENT]  # 9: ROOT
    ]

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

    env = gym.make("Taxi-v3", render_mode="human" if render_training or not do_train else None)
    state_size = env.observation_space.n
    action_size = len(ACTIONS)

    vtable = np.zeros((action_size, state_size))
    C = np.zeros((action_size, state_size, action_size))

    if do_train:
        train(show_plot=show_plot)
        env.close()
        env = gym.make("Taxi-v3", render_mode="human")

    test()
    env.close()
