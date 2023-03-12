import gym


if __name__ == "__main__":
    
    env = gym.make("CartPole-v0")
    
    # Monitor what happen with a video
    env = gym.wrappers.Monitor(env, "recording", force=True)

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    env.close()
    env.env.close()
