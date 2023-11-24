import gymnasium as gym


def learning_step(env, state):
    action = env.action_space.sample()
    env.step(action)
    env.render()


def main():
    env = gym.make("Taxi-v3", render_mode="human")

    state = env.reset()

    num_steps = 99
    for s in range(num_steps + 1):
        print(f"\rstep: {s} out of {num_steps}", end="")
        learning_step(env, state)

    env.close()


if __name__ == '__main__':
    main()
