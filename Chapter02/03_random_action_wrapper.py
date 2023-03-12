import gym
from typing import TypeVar
import random

Action = TypeVar('Action')

# Example of action wrapper to modify what happens when an action is taken
class RandomActionWrapper(gym.ActionWrapper):
    
    def __init__(self, env, epsilon=0.5):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:

        # If we get a random number below epsilon then we return a random action
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        # Else we return the original action
        return action


if __name__ == "__main__":
    
    # We instantiate again the cart pole environment but with modified actions
    env = RandomActionWrapper(gym.make("CartPole-v0"))

    # We make it start from the beginning
    obs = env.reset()
    total_reward = 0.0

    while True:
        # It happens in the background, here we always take action 0
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print("Reward got: %.2f" % total_reward)
