import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# Cross Entropy can learn only at the end of the episodes

HIDDEN_SIZE = 128
BATCH_SIZE = 16 # number of episodes per iteration
PERCENTILE = 70 # to pick best episodes based on reward


# Network that given an observation gives the prob. distribution over actions
# actually we do not have the softmax included as we will use CrossEntropyLoss
# It is train to get the probability observed in the best episodes
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# They are sort of dictionaries
# The first will save observation and action for every step in the episode
# The second will store the total undiscounted reward and all steps (list)
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# This generates batches with episodes given the environment, NN and N. episodes
def iterate_batches(env, net, batch_size):
    
    batch = [] # List of Episode elements
    episode_reward = 0.0 # Keep track of the total reward of the episode
    episode_steps = [] # List of EpisodeStep elements
    obs = env.reset() # Initialize the observation
    softmax = nn.Softmax(dim=1)
    
    while True:
        # We select the actions based on the output of the NN so that the
        # slection improves over time

        # For this observation create a tensor and feed it to the NN
        obs_v = torch.FloatTensor([obs])
        act_probs_v = softmax(net(obs_v))
        # Get the prob. distribution an select an action
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        # With this action take a step
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        # Save the old observation and the action taken from there
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        # If the episode is over
        if is_done:
            # Save the total reward/return and all steps
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            # Reinitialise the episode related variables, not batch!
            episode_reward = 0.0
            episode_steps = []
            # Reset the environment to start a new episode
            next_obs = env.reset()
            # If we have reached the amount of episodes we need we yield
            if len(batch) == batch_size:
                yield batch
                batch = []
        # If we are not done we use the latest observation and keep on
        obs = next_obs

# Core of the cross entropy method
def filter_batch(batch, percentile):

    # For batch we need to get the total rewards and compute the percentile
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    # Only if the reward is higher than the defined percentile we keep its steps
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        # We save both observations and actions for training
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    # They are transformed into tensors so that they are ready for training
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":

    # Initialise the env, get the observations space size and actions space size
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="monitoring", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize the network, loss, optimizer and writer
    # The net goes from observ to prob action in the BEST EPISODES
    net = Net(obs_size, HIDDEN_SIZE, n_actions) 
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    # Get a number of full episodes (BATCH_SIZE of them)
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        # iter_no is a sequential number, batch is a list of size 16

        # Filter the best episodes based on PERCENTILE
        # Get the observations, the actions, bound and mean reward
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        
        # Backpropagate considering the BEST ESPISODES
        optimizer.zero_grad()
        action_scores_v = net(obs_v) # Feed the observations
        loss_v = objective(action_scores_v, acts_v) # Compare the actions
        loss_v.backward()
        optimizer.step()

        # Print and save info for tensorboard monitoring
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        
        # Close if the mean reward is high enough
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
