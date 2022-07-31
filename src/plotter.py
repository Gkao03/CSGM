import numpy as np
import matplotlib.pyplot as plt


def plot_multiple(xs, ys, legends, title="title", xlabel="x label", ylabel="y label", savefile="plot.png", ylim=None):
    for x, y, legend in zip(xs, ys, legends):
        plt.plot(x, y, label=legend)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.savefig(savefile)
    print("saved plot")
    plt.close()


def get_data(fpath):
    with open(fpath, 'r') as fp:
        data = [line.rstrip('\n') for line in fp]

    data = [line.split(" ") for line in data]
    measurements = [int(line[0]) for line in data]
    recon_error = [float(line[1]) for line in data]
    recon_pixel_error = [float(line[2]) for line in data]

    return measurements, recon_error, recon_pixel_error


if __name__ == '__main__':
    fpaths = [
        "/media/biometrics/interns2022/gorek/csgm/out/dcgan_recon.txt",
        "/media/biometrics/interns2022/gorek/csgm/out/lassodb1_recon.txt",
        "/media/biometrics/interns2022/gorek/csgm/out/lassodct_recon.txt",
    ]

    legends = [
        "DCGAN+Reg",
        "Lasso (Wavelet)",
        "Lasso (DCT)"
    ]

    xs = []
    ys = []

    for fpath in fpaths:
        measurements, _, recon_pixel_error =  get_data(fpath)
        xs.append(measurements)
        ys.append(recon_pixel_error)

    assert len(legends) == len(xs) == len(ys)

    title = "CSGM: DCGAN and LASSO Reconstruction"
    xlabel = "Measurements"
    ylabel = "Recon Error (per pixel)"
    savefile = "/media/biometrics/interns2022/gorek/csgm/out/plot.png"

    plot_multiple(xs, ys, legends, title=title, xlabel=xlabel, ylabel=ylabel, savefile=savefile)
