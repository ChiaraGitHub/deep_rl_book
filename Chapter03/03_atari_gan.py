#!/usr/bin/env python
import random
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import gym
import gym.spaces

import numpy as np

# Initialise parameters
log = gym.logger
log.set_level(gym.logger.INFO)
LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16
IMAGE_SIZE = 64 # dimension input image will be rescaled
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100 # For logging and adding to tensorboard
SAVE_IMAGE_EVERY_ITER = 1000 # For tensorboard


# Observation Wrapper to preprocess the images
class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to the first place
    3. set type to float32
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
                                        self.observation(old_space.low),
                                        self.observation(old_space.high),
                                        dtype=np.float32)

    # method to modify the observation
    def observation(self, observation):
        # resize image so that it is a square
        new_obs = cv2.resize(
                            src=observation,
                            dsize=(IMAGE_SIZE, IMAGE_SIZE)
                            )
        # move axis so that the color channel comes first
        new_obs = np.moveaxis(new_obs, 2, 0)
        # make it of type float32
        return new_obs.astype(np.float32)

# Discriminator / Detective: is the image real or not?
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        # it takes the image as input and outputs the prob of being real image
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=DISCR_FILTERS,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS,
                      out_channels=DISCR_FILTERS * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2,
                      out_channels=DISCR_FILTERS * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4,
                      out_channels=DISCR_FILTERS * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass batch
        conv_out = self.conv_pipe(x)
        # Return probs for every image in batch
        return conv_out.view(-1, 1).squeeze(dim=1)

# Generator / Cheater: Generate fake images
class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves random input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE,
                               out_channels=GENER_FILTERS * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8,
                               out_channels=GENER_FILTERS * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4,
                               out_channels=GENER_FILTERS * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2,
                               out_channels=GENER_FILTERS,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS,
                               out_channels=output_shape[0],
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

# Get normalised observations
def iterate_batches(envs, batch_size=BATCH_SIZE):
    """
    For environement get a number of observations equal to the batch size
    """
    # set environemnt to initial conditions
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        # Get a random environment
        e = next(env_gen)
        print(e)
        # Step and get observation of size (3, 64, 64)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01: # Exclude observations with zero mean
            batch.append(obs)
        # Once the batch size is reached normalize and return
        if len(batch) == batch_size: 
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true',
        help="Enable cuda computation")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # List of environments instances via the wrapper
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
            ]
    # The input shape is always (3, 64, 64) because of the wrapper
    input_shape = envs[0].observation_space.shape
    
    # Instantiate networks
    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    # Define loss and optimizers
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
                            params=net_gener.parameters(),
                            lr=LEARNING_RATE,
                            betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
                            params=net_discr.parameters(),
                            lr=LEARNING_RATE,
                            betas=(0.5, 0.999))
    
    # Create writer for TensorBoard
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    # The function "iterate_batches"  picks each time a random environment and 
    # steps it until the total amount of observations equals the batch size
    for batch_v in iterate_batches(envs):
        
        # batch_v has shape (16, 3, 64, 64)

        # First GENERATE
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1) # Random input
        gen_input_v = gen_input_v.to(device)
        gen_output_v = net_gener(gen_input_v)
        # gen_output_v has shape (16, 3, 64, 64)

        # Train discriminator --------------------------------------------------
        dis_optimizer.zero_grad()
        batch_v = batch_v.to(device)
        dis_output_true_v = net_discr(batch_v) # Real images
        dis_output_fake_v = net_discr(gen_output_v.detach()) # Fake images
        # Detach as the grads of this pass should not flow into the generator

        # We want the discriminator to undestand if they are real or fake
        # This is a sum of 2 scalars
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # Train generator ------------------------------------------------------
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        # dis_output_v has shape 16, one label per image
        
        # We want the generator to make the discriminator think that the images
        # are real: output of discriminator ~= true_labels==1
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1

        # Write to monitor in Tensorboard
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(dis_losses))
            writer.add_scalar(
                "gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        
        # Save some images
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)
