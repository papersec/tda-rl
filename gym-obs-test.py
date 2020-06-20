import gym, time

if __name__ == '__main__':
    env = gym.make('Breakout-v0').unwrapped

    print(env.render(mode='rgb_array').dtype)

    time.sleep(5)
