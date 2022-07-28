from sklearn.linear_model import Lasso
import numpy as np
import copy
import os
import pywt
from PIL import Image
from torchvision import transforms
from utils import *


class Args:
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                             transforms.CenterCrop(64),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std=[0.5, 0.5, 0.5])])
        self.wavelet_level = 0
        self.np_image_shape = (64, 64, 3)
        self.torch_image_shape = (3, 64, 64)
        self.image_size = 64
        self.image_path = "/media/FiveTB/oosto_weapon2022/gorek/data/celeba/img_align_celeba/051662.jpg"
        self.out_image_dir = "/media/biometrics/interns2022/gorek/csgm/imgs/051662_lasso"
        self.out_dir = "/media/biometrics/interns2022/gorek/csgm/out"
        # self.alpha = 0.000000001
        self.alpha = 0.00001
        # self.learning_rate = 0.01
        self.num_iters = 1000


def wave_coeffs_to_array(coeffs):
    """
    https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html
    """
    levels = len(coeffs) - 1
    cA = coeffs[0]
    res_array = np.copy(cA)

    for cH, cV, cD in coeffs[1:]:
        assert cH.shape == cV.shape == cD.shape
        # print(f"cH shape: {cH.shape}")
        temp_array = np.copy(res_array)
        # print(f"temp shape: {temp_array.shape}")
        x, y = temp_array.shape
        res_array = np.zeros((2 * x, 2 * y))
        # print(f"res array shape: {res_array.shape}")

        # set LL
        res_array[:x, :y] = temp_array
        # set LH
        res_array[:x, y:] = cH
        # set HL
        res_array[x:, :y] = cV
        # set HH
        res_array[x:, y:] = cD

    return res_array


def array_to_wave_coeffs(array, level, image_size):
    curr_size = image_size // (2 ** level)
    cA = array[:curr_size, :curr_size]
    coeffs = []
    coeffs.append(cA)

    for _ in range(level):
        cH = array[:curr_size, curr_size:curr_size*2]
        cV = array[curr_size:curr_size*2, :curr_size]
        cD = array[curr_size:curr_size*2, curr_size:curr_size*2]
        assert cH.shape == cV.shape == cD.shape

        coeffs.append((cH, cV, cD))
        curr_size *= 2

    return coeffs


def run_lasso(X, y, alpha, max_iter, tol=0.001):
    """
    arguments and algorithm according to
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """
    lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    lasso.fit(X=X, y=y)

    assert lasso.coef_.shape == (args.image_size ** 2,)

    return lasso.coef_


def main(args: Args):
    measurements = [20, 50, 100, 200, 500, 1000, 2500, 5000, 7500, 10000]
    recon_errors = []
    recon_pixel_errors = []

    for m in measurements:
        X = get_original_image_tensor(args.image_path, args.transform)
        assert X.shape == args.torch_image_shape, f"shape should be {args.torch_image_shape}"
        X_splits = torch.split(X, 1, dim=0)
        xhats = []

        for X_split in X_splits:
            X_np = to_data(X_split.squeeze())
            X_wave_coeff_list = pywt.wavedec2(X_np, wavelet='db1', mode='zero', level=args.wavelet_level)
            X_wavelet_array = wave_coeffs_to_array(X_wave_coeff_list)
            assert X_wavelet_array.shape == (args.image_size, args.image_size)
            
            A = to_data(A_measurement(m=m, n=X_wavelet_array.size))
            n = to_data(noise(m=m, standard_dev=0.0001))
            Y = A @ X_wavelet_array.flatten() + n

            wavelet_vector = run_lasso(X=A, y=Y, alpha=args.alpha, max_iter=args.num_iters)
            wavelet_array = wavelet_vector.reshape((args.image_size, args.image_size))
            wavelet_coeffs = array_to_wave_coeffs(wavelet_array, level=args.wavelet_level, image_size=args.image_size)
            xhat = pywt.waverec2(wavelet_coeffs, wavelet='db1', mode='zero')
            assert xhat.shape == (args.image_size, args.image_size)

            xhats.append(xhat)
        
        final_xhat = np.stack(xhats, axis=0).transpose(1, 2, 0)
        final_xhat_uint8 = scale_array_uint8(final_xhat)
        print(final_xhat_uint8)
        np_to_image_save(final_xhat_uint8, os.path.join(args.out_image_dir, f"xhat_{m}.png"))

        break


if __name__ == '__main__':
    args = Args()
    main(args)