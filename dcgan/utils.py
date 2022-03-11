import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, ConcatDataset
import os
import random


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
