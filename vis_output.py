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


def show_img(img_name):
    img = cv.imread(img_name)
    print(type(img))
    print(np.shape(img))
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
    val_file_names = glob.glob('dataset/test_logic/images/*.png')
    val_dataset = LoadDatasetVal(val_file_names)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=16)

    weight_load = 'Logs/T20200911_211620/weights_659.pth'
    device = torch.device("cuda")
    model = LWANet(num_classes=8, pretrained=True)

    model.final[0] = nn.Sequential(
        nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0),
        nn.Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0))

    model.final[1] = nn.Sequential(
        nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0),
        nn.Conv2d(24, 8, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0))

    model.eval()
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=device).items()})
    model = model.cuda(0)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.cuda(0)
            outputs_ln = model(inputs)
            outputs_exp = torch.exp(outputs_ln)
            output_max = outputs_exp.argmax(dim=1)
            output_max = output_max.cpu().numpy()

            output_maxs = np.squeeze(output_max, axis=0)
            output_maxs = output_maxs.astype('float32')
            vis2 = cv.cvtColor(output_maxs, cv.COLOR_GRAY2BGR)
            cv.imshow('output', vis2)

            inputs_np = inputs.cpu().numpy()
            inputs_np = ((inputs_np * 0.5) + 0.5) * 255
            inputs_np = np.squeeze(inputs_np, 0)
            inputs_np = inputs_np.transpose(1, 2, 0)


            #'''
            data = np.zeros((136, 240, 3)) + 255
            key1 = output_max[0] != 0

            data[:, :, 0][key1] = 0
            data[:, :, 1][key1] = 0
            data[:, :, 2][key1] = 150

            dim = (960, 544)
            resized_data = cv.resize(data, dim, interpolation=cv.INTER_AREA)
            cv.imshow("image", resized_data)
            #'''
            inputs_np = cv.cvtColor(inputs_np, cv.COLOR_RGB2BGR)

            stack_img = cv.addWeighted(inputs_np, 0.8, resized_data, 0.2, 0.4, dtype=cv.CV_8UC3)
            cv.imshow("image", stack_img)
            k = cv.waitKey(0)
            if k == 27:
                cv.destroyAllWindows()


if __name__ == '__main__':
    img_file = '/home/yanjun/LWANet/dataset/train_MICCAI2019/labels/31360.png'

    #img_file = 'dataset/test_high_res_pic/images/*.png'
    #loop_all_imgs(img_file)

    predict_image()
