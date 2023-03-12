#!/usr/bin/env python
import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


# Observation Wrapper to preprocess the images
class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
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

# Discriminator / Detective
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

# Generator / Cheater
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
        # Step and get observation of size (3, 64, 64)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
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
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
            ]
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

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    def process_batch(trainer, batch):
        
        # First generate
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch.to(device)
        gen_output_v = net_gener(gen_input_v)

        # Train discriminator ---------
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        # Detach as the grads of this pass should not flow into the generator

        # We want the discriminator to undestand if they are real or fake
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()

        # Train generator -----------
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        
        # We want the generator to make the discriminator think that the images
        # are real
        gen_loss = objective(dis_output_v, true_labels_v)
        gen_loss.backward()
        gen_optimizer.step()

        # This part is different! Save images
        if trainer.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
            fake_img = vutils.make_grid(
                gen_output_v.data[:64], normalize=True)
            trainer.tb.writer.add_image(
                "fake", fake_img, trainer.state.iteration)
            real_img = vutils.make_grid(
                batch_v.data[:64], normalize=True)
            trainer.tb.writer.add_image(
                "real", real_img, trainer.state.iteration)
            trainer.tb.writer.flush()
        
        # Return data to be tracked during training
        return dis_loss.item(), gen_loss.item()

    # We pass to engine the function to process
    engine = Engine(process_batch)

    # We attach to engine some functions that we would likebe called
    tb = tb_logger.TensorboardLogger(log_dir=None)
    engine.tb = tb
    
    # Attach the running average transformations for the losses
    RunningAverage(output_transform=lambda out: out[1]).\
        attach(engine, "avg_loss_gen")
    RunningAverage(output_transform=lambda out: out[0]).\
        attach(engine, "avg_loss_dis")

    # The loss averages will be written in TensorBoard at every iteration
    handler = tb_logger.OutputHandler(
                tag="train",
                metric_names=['avg_loss_gen', 'avg_loss_dis'])
    tb.attach(engine, log_handler=handler,
              event_name=Events.ITERATION_COMPLETED)

    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(trainer):

        # Write a log
        if trainer.state.iteration % REPORT_EVERY_ITER == 0:
            log.info("%d: gen_loss=%f, dis_loss=%f",
                     trainer.state.iteration,
                     trainer.state.metrics['avg_loss_gen'],
                     trainer.state.metrics['avg_loss_dis'])
    
    # we pass the data to run
    engine.run(data=iterate_batches(envs))
