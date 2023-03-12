#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # Rewards has as key state, action, next_state and as value the rewards
        self.rewards = collections.defaultdict(float)
        # Dict with key state & action and as value a dict with key new_state 
        # and as value num. visits
        self.transits = collections.defaultdict(collections.Counter)
        # Dict with states are keys and values as values
        self.values = collections.defaultdict(float)

    # We play n steps to update the rewards and transits tables
    def play_n_random_steps(self, count):
        for _ in range(count):
            # We get a random action, step, observe and update the tables
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    # Calculate the value of a state and action pair
    def calc_action_value(self, state, action):
        # Get the transition: it is a dict with new_state and num. visits
        target_counts = self.transits[(state, action)]
        # We take the sum of the visits over all successor states
        total = sum(target_counts.values())
        action_value = 0.0
        # For every next state and count pair we get the reward, compute the 
        # discounted sum and scale it by the percentage of visits (probability
        # of this transition as in the Bellman equations)
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    # To determine which action to take based on the value given the state
    # We take the action that maximises the value
    def select_action(self, state):
        best_action, best_value = None, None
        # For that state and every action we calculate the value
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # Play full episode taking the best actions and using another env so that
    # we don't mess up with the state of the main one
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # Value iteration to update the states values
    def value_iteration(self):
        # For every state in the environment
        for state in range(self.env.observation_space.n):
            # Compute the action value for every action possible
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            # Take the max value and store it
            self.values[state] = max(state_values)


if __name__ == "__main__":
    
    # Initialize environment, agent and summary writer
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        
        # We play random steps and run the value iteration to update the values
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # We run some test episodes using the current values and write to 
        # Tensorboard
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        
        # We stop if the reward is high enough
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
