import random
from typing import List

# Basic example of environment
class Environment:

    # The environment provides observations, possible actions and rewards
    
    def __init__(self):
        # We git it a maximum duration
        self.steps_left = 10

    def get_observation(self) -> List[float]:
        # The observation vector has length 3
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        # We have 2 possible actions
        return [0, 1]

    def is_done(self) -> bool:
        # If there are no steps left we are done
        return self.steps_left == 0

    def action(self, action: int) -> float:
        # If we are done we raise a "Game over"
        if self.is_done():
            raise Exception("Game is over")
        # Else we decrease the steps left
        self.steps_left -= 1
        # We return a random reward
        return random.random()

# Basic example of agent
class Agent:
    # The agent needs to step given the environment

    def __init__(self):
        # We initialise the total reward
        self.total_reward = 0.0

    def step(self, env: Environment):
        # Get all observations
        current_obs = env.get_observation()
        # Get all actions
        actions = env.get_actions()
        # Get a reward based on a random action
        reward = env.action(random.choice(actions))
        # Add the reward to the total rewards
        self.total_reward += reward


if __name__ == "__main__":

    # Instantiate environment and agent
    env = Environment()
    agent = Agent()

    # Iterate until we are done
    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
