from sklearn.linear_model import Lasso
from scipy.fft import dctn, idctn
import numpy as np
import copy
import os
import pywt
from PIL import Image
from torchvision import transforms
from utils import *

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


class Args:
    def __init__(self):
        self.method = "wavelet"
        self.transform = transforms.Compose([transforms.Resize((64, 64)),
                                             transforms.CenterCrop(64),
                                             transforms.ToTensor(),])
                                            #  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            #                       std=[0.5, 0.5, 0.5])])
        self.wavelet_level = 6
        self.np_image_shape = (64, 64, 3)
        self.torch_image_shape = (3, 64, 64)
        self.image_size = 64
        self.image_path = "/media/FiveTB/oosto_weapon2022/gorek/data/celeba/img_align_celeba/051662.jpg"
        self.out_image_dir = "/media/biometrics/interns2022/gorek/csgm/imgs"
        self.out_dir = "/media/biometrics/interns2022/gorek/csgm/out"
        # self.alpha = 0.000000000001
        self.alpha = 0.000000000001
        # self.alpha = 0.00001
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


# another method to wave_coeffs_to_array (should be the same result)
def wave_coeffs_to_array2(coeffs):
    temp = [coeffs[0].flatten()]
    for cH, cV, cD in coeffs[1:]:
        temp.append(cH.flatten())
        temp.append(cV.flatten())
        temp.append(cD.flatten())

    res = np.concatenate(temp, axis=-1)
    return res


# another method to array_to_wave_coeffs (should be the same result)
def array_to_wave_coeffs2(array, level, image_size):
    curr_size = image_size // (2 ** level)
    curr_idx = curr_size ** 2
    coeffs = []

    coeffs.append(np.copy(array[:curr_idx]).reshape(curr_size, curr_size))

    for _ in range(level):
        length = curr_size ** 2
        
        cH = np.copy(array[curr_idx:curr_idx + length]).reshape(curr_size, curr_size)
        
        curr_idx = curr_idx + length
        cV = np.copy(array[curr_idx:curr_idx + length]).reshape(curr_size, curr_size)

        curr_idx = curr_idx + length
        cD = np.copy(array[curr_idx:curr_idx + length]).reshape(curr_size, curr_size)

        coeffs.append((cH, cV, cD))
        curr_idx = curr_idx + length
        curr_size *= 2

    return coeffs


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


def run_lasso(X, y, alpha, max_iter, tol=0.0001):
    """
    arguments and algorithm according to
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """
    lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    lasso.fit(X=X, y=y)

    assert lasso.coef_.shape == (args.image_size ** 2,)

    return lasso.coef_


def main(args: Args):
    measurements = [20, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
    recon_errors = []
    recon_pixel_errors = []

    for m in measurements:
        X = get_original_image_tensor(args.image_path, args.transform)
        assert X.shape == args.torch_image_shape, f"shape should be {args.torch_image_shape}"
        X_splits = torch.split(X, 1, dim=0)
        xhats = []

        for X_split in X_splits:
            X_np = to_data(X_split.squeeze())
            # X_np = scale_array_uint8(X_np)

            if args.method == "wavelet":
                X_wave_coeff_list = pywt.wavedec2(X_np, wavelet='db1', mode='periodic', level=args.wavelet_level)
                X_wavelet_array = wave_coeffs_to_array2(X_wave_coeff_list)  # vectorize
                X_res = X_wavelet_array
                # assert X_wavelet_array.shape == (args.image_size, args.image_size)
            elif args.method == "dct":
                X_dct_array = dctn(X_np)
                X_res = X_dct_array
            else:
                raise ValueError(f"Method {args.method} not supported!")

            
            A = to_data(A_measurement(m=m, n=X_res.size))
            n = to_data(noise(m=m, standard_dev=0.0001))
            Y = A @ X_res.flatten() + n

            if args.method == "wavelet":
                wavelet_vector = run_lasso(X=A, y=Y, alpha=args.alpha, max_iter=args.num_iters)
                wavelet_coeffs = array_to_wave_coeffs2(wavelet_vector, level=args.wavelet_level, image_size=args.image_size)
                # wavelet_array = wavelet_vector.reshape((args.image_size, args.image_size))
                # wavelet_coeffs = array_to_wave_coeffs(wavelet_array, level=args.wavelet_level, image_size=args.image_size)
                xhat = pywt.waverec2(wavelet_coeffs, wavelet='db1', mode='periodic')
                assert xhat.shape == (args.image_size, args.image_size), f"got shape {xhat.shape}"
            elif args.method == "dct":
                dct_arr = run_lasso(X=A, y=Y, alpha=args.alpha, max_iter=args.num_iters).reshape(args.image_size, args.image_size)
                xhat = idctn(dct_arr)
            else:
                raise ValueError(f"Method {args.method} not supported!")

            xhats.append(xhat)

        final_xhat = np.stack(xhats, axis=0)
        final_xhat = scale_array(final_xhat, 0, 1)
        recon_error = get_recon_error(torch.tensor(final_xhat), X)
        recon_errors.append(recon_error)
        recon_pixel_error = get_recon_error_per_pixel(torch.tensor(final_xhat), X)
        recon_pixel_errors.append(recon_pixel_error)

        print(f"Measurement: {m} | Recon Error: {recon_error} | Recon Pixel Error: {recon_pixel_error}")

        final_xhat = final_xhat.transpose(1, 2, 0)
        np_to_image_save(final_xhat, os.path.join(args.out_image_dir, f"xhat_{m}.png"))

    plot(measurements, recon_pixel_errors, title="LASSO", xlabel="# Measurements", ylabel="Recon Error (per pixel)", xticks=False, save_path="./plot.png")  # os.path.join(args.out_image_dir, "plot.png"))
    # save_list_values(os.path.join(args.out_dir, "lassodct_recon.txt"), measurements, recon_errors, recon_pixel_errors)


if __name__ == '__main__':
    args = Args()
    main(args)
