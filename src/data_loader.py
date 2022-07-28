import glob
import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
import diff_augment


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        tensor_image = tensor_image * 2 - 1
        return tensor_image, 0.


def get_data_loader(data_path, opts):
    """Creates training and test data loaders.
    """
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'deluxe':
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        # transforms.Resize(osize, Image.BICUBIC)
        # transforms.RandomCrop(opts.image_size)
        # transforms.RandomHorizontalFlip()
        transform_list = [transforms.Resize(osize, Image.BICUBIC), 
                          transforms.RandomCrop(opts.image_size), 
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]
        train_transform = transforms.Compose(transform_list)

    # dataset = CustomDataSet(os.path.join('data/', data_path), opts.ext, train_transform)
    dataset = CustomDataSet(data_path, opts.ext, train_transform)
    dloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return dloader

def get_celeba_dataloader(opts, img_crop_size=64):
    celeba_train_transform = transforms.Compose([transforms.Resize((img_crop_size, img_crop_size)),
                                                 transforms.CenterCrop(img_crop_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                      std=[0.5, 0.5, 0.5])])

    celeba_dset = datasets.ImageFolder(root=opts.data, transform=celeba_train_transform)
    celeba_dataloader = DataLoader(dataset=celeba_dset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)

    return celeba_dataloader
