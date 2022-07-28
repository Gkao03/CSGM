# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import argparse
import os
from random import sample
import warnings

import imageio
from transformers import get_scheduler

warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np

# Torch imports
import torch
import torch.optim as optim
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

# Local imports
import utils
from data_loader import get_celeba_dataloader, get_data_loader
from models import DCGenerator, DCDiscriminator

from diff_augment import DiffAug
policy = 'color,translation,cutout'


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D):
    """Prints model information for the generators and discriminators.
    """
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = DCDiscriminator(conv_dim=opts.conv_dim, norm=opts.conv_norm)

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_iter%d.pkl' % iteration)
    D_path = os.path.join(opts.checkpoint_dir, 'D_iter%d.pkl' % iteration)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    # print('Saved {}'.format(path))


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))

    path = os.path.join(opts.sample_dir, '{:s}-{:06d}.png'.format(name, iteration))
    imageio.imwrite(path, grid)
    # print('Saved {}'.format(path))


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    # device
    device = utils.get_device()

    # fake and real labels
    real_label = 1
    fake_label = 0

    # diffaug
    policy = 'color,translation,cutout'
    diffaug_transforms = DiffAug(policy=policy)

    # Create generators and discriminators
    G, D = create_model(opts)

    # criterion
    criterion = nn.BCELoss()

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # lr schedulers if needed
    # g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=2, gamma=0.5, verbose=True)
    # d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=2 ,gamma=0.5, verbose=True)

    # Generate fixed noise for sampling from the generator
    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)  # batch_size x noise_size x 1 x 1

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    for epoch in range(opts.num_epochs):

        for i, batch in enumerate(train_dataloader):

            # real_images, labels = batch
            # real_images, labels = utils.to_var(real_images), utils.to_var(labels).long().squeeze()
            real_images = batch[0]
            real_images = utils.to_var(real_images)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################
            D.zero_grad()

            # 1. Compute the discriminator loss on real images
            # D_real_loss = torch.mean((D(real_images) - 1)**2)
            if opts.use_diffaug:
                real_images = diffaug_transforms(real_images)

            label = torch.full((opts.batch_size,), real_label, dtype=torch.float, device=device)
            output = D(real_images).view(-1)
            D_real_loss = criterion(output, label)
            D_real_loss.backward()
            D_x = output.mean().item()

            # 2. Sample noise
            # noise = sample_noise(batch_size=opts.batch_size, dim=opts.noise_size)
            noise = torch.randn(opts.batch_size, opts.noise_size, 1, 1, device=device)

            # 3. Generate fake images from the noise
            fake_images = G(noise)
            label.fill_(fake_label)

            # 3.1 differentiable augmentation
            if opts.use_diffaug:
                fake_images = diffaug_transforms(fake_images)

            # 4. Compute the discriminator loss on the fake images
            # D_fake_loss = torch.mean((D(fake_images.detach())) ** 2)
            output = D(fake_images.detach()).view(-1)
            # D_fake_loss = torch.mean((D(fake_images.detach())) ** 2)
            D_fake_loss = criterion(output, label)
            D_fake_loss.backward()
            D_G_z1 = output.mean().item()

            # D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_total_loss = D_fake_loss + D_real_loss

            # update the discriminator D
            # d_optimizer.zero_grad()
            # D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake_images).view(-1)

            # 1. Sample noise
            # noise = sample_noise(batch_size=opts.batch_size, dim=opts.noise_size)

            # 2. Generate fake images from the noise
            # fake_images = G(noise)

            # 2.1 differentiable augmentation
            if opts.use_diffaug:
                fake_images = diffaug_transforms(fake_images)

            # 3. Compute the generator loss
            # G_loss = torch.mean((D(fake_images) - 1) ** 2)
            G_loss = criterion(output, label)

            # update the generator G
            # g_optimizer.zero_grad()
            G_loss.backward()
            D_G_z2 = output.mean().item()
            g_optimizer.step()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Epoch [{}/{}] | Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       epoch, opts.num_epochs, iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
                # logger.add_scalar('D/fake', D_fake_loss, iteration)
                # logger.add_scalar('D/real', D_real_loss, iteration)
                # logger.add_scalar('D/total', D_total_loss, iteration)
                # logger.add_scalar('G/total', G_loss, iteration)

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, 'real')

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1

        # g_scheduler.step()
        # d_scheduler.step()


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    # dataloader = get_data_loader(opts.data, opts)
    dataloader = get_celeba_dataloader(opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--noise_size', type=int, default=100)
    parser.add_argument('--conv_norm', type=str, default='instance')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--use_diffaug', default=False, action='store_true')

    # Data sources
    parser.add_argument('--data', type=str, default='/media/FiveTB/oosto_weapon2022/gorek/data/celeba', help='The folder of the training dataset.')
    parser.add_argument('--data_preprocess', type=str, default='basic', help='data preprocess scheme [basic|deluxe]')
    parser.add_argument('--ext', type=str, default='*.jpg', help='Choose the file type of images to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join("/media/FiveTB/oosto_weapon2022/gorek", "checkpoints"))
    parser.add_argument('--sample_dir', type=str, default=os.path.join("/media/FiveTB/oosto_weapon2022/gorek", "samples"))
    parser.add_argument('--log_step', type=int , default=100)
    parser.add_argument('--sample_every', type=int , default=1000)
    parser.add_argument('--checkpoint_every', type=int , default=1000)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    # opts.sample_dir = os.path.join('output/', opts.sample_dir,
    #                                '%s_%s' % (os.path.basename(opts.data), opts.data_preprocess))
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    print(f"augmentation type: {opts.data_preprocess}")
    print(f"using diff aug? {opts.use_diffaug}")

    # if os.path.exists(opts.sample_dir):
    #     cmd = 'rm %s/*' % opts.sample_dir
    #     os.system(cmd)
    # logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)