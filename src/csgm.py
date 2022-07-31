import torch
import torch.optim as optim
import numpy as np
import copy
import os
from PIL import Image
from torchvision import transforms
from utils import *
from models import DCGenerator, DCDiscriminator


class Args:
    def __init__(self):
        self.G_path = "/media/FiveTB/oosto_weapon2022/gorek/checkpoints/G_iter23000.pkl"
        self.noise_size = 100
        self.conv_dim = 64
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                             transforms.CenterCrop(64),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std=[0.5, 0.5, 0.5])])
        self.np_image_shape = (64, 64, 3)
        self.torch_image_shape = (3, 64, 64)
        self.image_path = "/media/FiveTB/oosto_weapon2022/gorek/data/celeba/img_align_celeba/051662.jpg"
        self.out_image_dir = "/media/biometrics/interns2022/gorek/csgm/imgs"
        self.out_dir = "/media/biometrics/interns2022/gorek/csgm/out"
        # self.L_lambda = 0.001  # original
        self.L_lambda = 0.00000001
        self.learning_rate = 0.01
        self.num_iters = 500


def run_csgm(A, X, Y, z_init, lbda, step_size, num_iters, G_model, optimizer: optim.Optimizer):
    """
    minimize objective:
    || AG(z) - y ||^2 + lambda ||z||^2.
    Everything is done as torch tensor. must reshape to correct np array channels when done.
    Return zhat as torch tensor
    """
    G_model.eval()
    z = z_init
    G_z = G_model(z).detach()
    best_meas_error = get_measurement_error(G_z, A, Y)
    best_recon_error = get_recon_error(G_z, X)
    zhat = copy.deepcopy(z)

    for i in range(num_iters):
        G_model.train()
        optimizer.zero_grad()
        z.requires_grad_()  # turn on grad for tensor

        objective = get_objective_loss(G_model, A, Y, z, lbda)
        objective.backward()
        optimizer.step()
        gradient_z = z.grad
        z = z - step_size * gradient_z
        z = z.detach()

        G_model.eval()
        G_z = G_model(z).detach()
        meas_error = get_measurement_error(G_z, A, Y)
        recon_error = get_recon_error(G_z, X)

        if meas_error < best_meas_error:
            best_meas_error = meas_error
            zhat = copy.deepcopy(z)

        if recon_error < best_recon_error:
            best_recon_error = recon_error

        print(f"[{i+1}/{num_iters}] | Measurement Error: {meas_error} | Recon Error: {recon_error}")

    return zhat


def z_to_x(G_model, z):
    assert len(z.shape) == 4
    assert z.shape[0] == 1

    x = G_model(z)
    return x

def main(args: Args):
    measurements = [20, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
    recon_errors = []
    recon_pixel_errors = []
    z_init = (torch.randn(1, args.noise_size, 1, 1) * 0.1).requires_grad_()

    for m in measurements:
        G_model = DCGenerator(noise_size=args.noise_size, conv_dim=args.conv_dim)
        load_DG_weights(G_model=G_model, G_path=args.G_path, D_model=None, D_path=None)
        G_model.train()
        optimizer = optim.Adam(G_model.parameters(), args.learning_rate)

        X = get_original_image_tensor(args.image_path, args.transform)
        A = A_measurement(m=m, n=torch.numel(X))
        n = noise(m=m, standard_dev=0.0001)
        # z_init = torch.randn(1, args.noise_size, 1, 1, requires_grad=True)
        Y = torch.matmul(A, X.flatten()) + n

        zhat = run_csgm(A, X, Y, z_init, args.L_lambda, 0.01, args.num_iters, G_model, optimizer)

        G_model.eval()
        G_z = G_model(zhat).detach()
        G_z_scale = torch.tensor(scale_array(G_z.numpy(), 0, 1))
        X_scale = torch.tensor(scale_array(X.numpy(), 0, 1))

        recon_error = get_recon_error(G_z_scale, X_scale)
        recon_errors.append(recon_error)
        recon_pixel_error = get_recon_error_per_pixel(G_z_scale, X_scale)
        recon_pixel_errors.append(recon_pixel_error)

        print(f"Measurement: {m} | Recon Error: {recon_error} | Recon Pixel Error: {recon_pixel_error}")

        xhat = z_to_x(G_model, zhat).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        np_to_image_save(xhat, os.path.join(args.out_image_dir, f"xhat_{m}.png"))

    plot(measurements, recon_pixel_errors, title="CSGM", xlabel="# Measurements", ylabel="Recon Error (per pixel)", xticks=False, save_path="./plot.png")
    # save_list_values(os.path.join(args.out_dir, "dcgan_recon.txt"), measurements, recon_errors, recon_pixel_errors)


if __name__ == '__main__':
    args = Args()
    main(args)
