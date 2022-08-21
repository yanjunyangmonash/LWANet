import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import PIL.Image as Image
from os.path import splitext
from os import listdir
import logging
import glob
import torchvision.transforms as transforms
import torch.nn.functional as F


class LoadDataset(Dataset):
    def __init__(self, imgs_dir, transform=None, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = imgs_dir.replace('image', 'label')
        self.transform = transform
        self.scale = scale
        # https://blog.finxter.com/python-one-line-for-loop-with-if/, if condition inside for loop
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mean=0.5, std=0.5):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        # Mask pre-process
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # numpy array HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # Not do normalize for labels
        if img_trans.max() > 1 and img_trans.shape[0] != 1:
            img_trans = img_trans / 255
            # Normalise data to [-1, 1]
            # https://discuss.pytorch.org/t/understanding-transform-normalize/21730/34?page=2
            img_trans = (img_trans - mean) / std
        # Add a channel here to use batchgenerator
        img_trans = np.expand_dims(img_trans, axis=0)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob.glob(self.masks_dir + '/' + idx + '.' + '*')
        img_file = glob.glob(self.imgs_dir + '/' + idx + '.' + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])  # Use [0] because mask_file is a list
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        data_dict = {'data': img, 'seg': mask}  # Numpy array
        if self.transform is not None:
            data_dict = self.transform(**data_dict)
        return data_dict


class LoadDatasetVal(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        ori_image = load_image(img_file_name)
        # (C, H, W)
        # After dataset pre-process, image size is reduced to half for faster calculation
        image = x_transforms(ori_image)

        mask = load_mask(img_file_name)
        mask = mask[np.newaxis, :, :]
        labels = torch.from_numpy(mask).float()
        labels = F.max_pool2d(labels, kernel_size=(4, 4), stride=4, padding=0)
        labels = labels.squeeze()
        return image, labels


x_transforms = transforms.Compose([
    transforms.ToTensor(),  # Automatically scale pixel value to [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def load_image(path):
    img_x = Image.open(path)
    return img_x


def load_mask(path):
    new_path = path.replace('image', 'label')
    mask = cv2.imread(new_path, 0)
    return mask.astype(np.uint8)

