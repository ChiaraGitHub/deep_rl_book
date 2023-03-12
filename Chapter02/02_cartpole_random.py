import gym


if __name__ == "__main__":

    # Make cart pole environment
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        
        # Take random action
        action = env.action_space.sample()
        # Step with this action
        obs, reward, done, _ = env.step(action)
        # Increase reward and steps counters
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
