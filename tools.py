import cv2 as cv
import glob
import torch
from LWANet import LWANet
from load_dataset_train import LoadDatasetVal
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
import PIL.Image as Image


def demo_show_color_mapping():
    data = np.zeros((3, 2, 2)) + 255
    array = np.zeros((2, 2))
    array[0][1] = 40
    # print(array)
    key1 = array == 40
    print(key1)
    print(data[0, :, :])
    data[0, :, :][key1] = 0
    print(data)


def check_mask_dim(img_name, draw=False):
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    img_pil = Image.open(img_name)
    img_nd = np.array(img_pil)
    print('mask image shape ', img_nd.shape)
    print('mask max value ', img_nd.max())
    print('class in mask ', np.unique(img_nd))

    if draw:
        cv.imshow('img', img)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()


def resize_img(img_name, dim=(1920, 1080), draw=False, mask=False):
    if mask:
        img = cv.imread(img_name, cv.IMREAD_UNCHANGED)
        print('Ori size ', img.shape)
        resized_img = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        cv.imwrite(img_name, resized_img)
    else:
        img = cv.imread(img_name)
        print('Ori size ', img.shape)
        resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        cv.imwrite(img_name, resized_img)
    print('New size ', resized_img.shape)
    if draw:
        cv.imshow('img', img)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()


def loop_all_imgs(file_dir, is_mask=False):
    # dir using 'dataset/test/images/*.png' format
    file_names = glob.glob(file_dir)
    for file in file_names:
        resize_img(file, dim=(960, 544), mask=is_mask)


def predict_image():
    val_file_names = glob.glob('dataset/test/images/*.png')
    val_dataset = LoadDatasetVal(val_file_names)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=16)

    weight_load = 'Logs/T20211130_121302/weights_182.pth'
    device = torch.device("cpu")
    model = LWANet(num_classes=2, pretrained=True)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=device).items()})
    model = model.cuda(0)
    model.eval()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.cuda(0)
            outputs_ln = model(inputs)
            outputs = torch.exp(outputs_ln[0])
            output_max = outputs.argmax(dim=0)
            output_max = output_max.cpu().numpy()
            outputs_tool =outputs[1].cpu().numpy()

            data = np.zeros((136, 240, 3)) + 255
            #key1 = output_max == 1
            key1 = outputs_tool > 0.57

            data[:, :, 0][key1] = 0
            data[:, :, 1][key1] = 255
            data[:, :, 2][key1] = 255

            dim = (960, 544)
            resized_data = cv.resize(data, dim, interpolation=cv.INTER_AREA)

            cv.imshow("image", resized_data)
            k = cv.waitKey(0)
            if k == 27:
                cv.destroyAllWindows()


if __name__ == '__main__':
    img_file = '/home/yanjun/LWANet/dataset/train_high_res_pic/labels/seq_1_frame000.png'
    check_mask_dim(img_file)

    #img_file = 'dataset/test_high_res_pic/images/*.png'
    #loop_all_imgs(img_file)

    # predict_image()