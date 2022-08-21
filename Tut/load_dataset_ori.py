import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import glob
from torch.utils.data import DataLoader

x_transforms = transforms.Compose([
    transforms.ToTensor(),  # Automatically scale pixel value to [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_trans = transforms.ToTensor()


class Load_Dataset(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        down_sample = 2
        img_file_name = self.file_names[idx]
        # ori_image size is 960*544
        ori_image = load_image(img_file_name)
        print(ori_image.size)
        # Data type to tensor and normalize the image
        image = x_transforms(ori_image)
        # Halve image size and change shape to CHW
        image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        print(image.shape)
        # Add two 0 columns to tensor_top and tensor_bottom, don't know why use this function
        image = F.pad(image, (0, 0, 2, 2), 'constant', 0)
        print(image.shape)

        mask = load_mask(img_file_name)
        mask = mask[np.newaxis, :, :]
        labels = torch.from_numpy(mask).float()
        print('mask size')
        print(labels.shape)
        labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        print(labels.shape)
        labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
        print(labels.shape)
        # Adjust label output to have a 1/4 size
        labels = F.max_pool2d(labels, kernel_size=(4, 4), stride=4, padding=0)
        print(labels.shape)
        labels = labels.squeeze()
        return image, labels


def load_image(path):
    img_x = Image.open(path)
    return img_x


def load_mask(path):
    new_path = path.replace('image', 'label')
    mask = cv2.imread(new_path, 0)
    # No idea when // 20
    mask = mask // 20

    return mask.astype(np.uint8)


if __name__ == '__main__':
    file_names = glob.glob('dataset/test/images/*.png')
    test_dataset = Load_Dataset(file_names)
    test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    # Check the number of data in the dataloader
    print(len(test_dataLoader.dataset))
    # Check the number of dataLoader for different batch_size
    print(len(test_dataLoader))
    # One pair in the dataloader
    it = iter(test_dataLoader)
    next(it)