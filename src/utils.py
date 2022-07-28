import os
import torch
import diff_augment
import imageio
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.detach().numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        print(f"making directory: {directory}")
        os.makedirs(directory)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_diffaug_tranforms():
    transform_list = []
    augments_dict = diff_augment.AUGMENT_FNS
    
    transform_list.extend(augments_dict['color'])
    transform_list.extend(augments_dict['translation'])
    transform_list.extend(augments_dict['cutout'])

    transform = transforms.Compose(transform_list)

    return transform


def np_to_image_save(np_array, path):
    imageio.imwrite(path, np_array)


def scale_array_uint8(arr):
    new_arr = ((arr - arr.min()) * (1/arr.ptp() * 255)).astype('uint8')
    return new_arr


def plot(x, y, title=None, xlabel=None, ylabel=None, xticks=True, save_path=None):
    plt.plot(x, y)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xticks:
        plt.xticks(x)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_list_values(out_path, *lists):
    with open(out_path, 'w') as output_file:
        for line in zip(*lists):
            output_file.write(" ".join(str(val) for val in line) + '\n')


def load_DG_weights(G_model, G_path, D_model, D_path):
    if D_model is not None and D_path is not None:
        D_model.load_state_dict(torch.load(D_path))

    if G_model is not None and G_path is not None:
        G_model.load_state_dict(torch.load(G_path))


# for 64x64x3 image, n=12288
def A_measurement(m, n):
    """
    Get measurement matrix A with m measurements.
    A is a random matrix with IID Gaussian entries with 0 mean
    and standard deviation of 1/m
    """
    mean = torch.zeros(m, n)
    std = torch.ones(m, n) * (1 / m)
    return torch.normal(mean=mean, std=std)


def noise(m, standard_dev):
    mean = torch.zeros(m)
    std = torch.ones(m) * standard_dev
    return torch.normal(mean=mean, std=std)


def get_original_image_tensor(image_path, transform):
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor


def get_objective_loss(G_model, A, Y, z, lbda, p=2):
    "|| AG(z) - y ||^2 + lambda ||z||^2"
    G_z = G_model(z)
    loss = torch.square(torch.norm(torch.matmul(A, G_z.flatten()) - Y, p=p)) + lbda * torch.square(torch.norm(z, p=p))
    return loss


def get_measurement_error(G_model, A, Y, z, p=2):
    G_z = G_model(z)
    meas_error = torch.square(torch.norm(torch.matmul(A, G_z.flatten()) - Y, p=p))
    return meas_error.item()


def get_recon_error(G_model, z, X, p=2):
    G_z = G_model(z)
    recon_error = torch.square(torch.norm(G_z.squeeze() - X, p=p))
    return recon_error.item()


def get_recon_error_per_pixel(G_model, z, X):
    G_z = G_model(z)
    abs_diff_tensor = torch.abs(G_z.squeeze() - X)
    mean_pixel_error = torch.mean(abs_diff_tensor)
    return mean_pixel_error.item()