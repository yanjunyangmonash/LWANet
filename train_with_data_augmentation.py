import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import tqdm
import datetime
import math
from validation import val_multi
import glob
from load_dataset_train import LoadDataset, LoadDatasetVal
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from focalloss import FocalLoss
import torch.nn as nn
from LWANet import LWANet
from LWANet import AFB
from torch._six import container_abcs
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms import NumpyToTensor, Compose


device_ids = [0]

parse = argparse.ArgumentParser()
num_classes = 5
new_num_classes = 5
lra = 0.001


def cat_data(catdata):
    return torch.cat(catdata, 0, out=None)


def new_collate(batch):
    # If data in batch is in dict
    if isinstance(batch[0], container_abcs.Mapping):
        return {key: cat_data([d[key] for d in batch]) for key in batch[0]}


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lra * (0.9 ** (epoch // 120))
    print('Updata lr, lr is ', str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_file_dir():
    train_file_dir = 'dataset/train_logic/images'
    val_file_names = glob.glob('dataset/test_logic/images/*.png')
    return train_file_dir, val_file_names


def train():
    mod = LWANet(num_classes=num_classes, pretrained=True)
    weight_load = 'Logs/T20211209_103331/weights_757.pth'
    # Loading model weight
    mod.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=torch.device("cpu")).items()})
    for param in mod.parameters():
        param.requires_grad = False
    # Modify final layer output

    mod.afb1 = AFB(24, 24)

    mod.final[0] = nn.Sequential(
        nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5))

    mod.final[1] = nn.Sequential(
        nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False),
        nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Conv2d(24, new_num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(new_num_classes),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5))

    #print(mod)
    model = mod.cuda(device_ids[0])

    batch_size = args.batch_size
    criterion = FocalLoss(alpha=0.15, gamma=5)
    optimizer = optim.Adam(model.parameters(), lr=lra)

    # Data Augmentation
    numpy_to_tensor = NumpyToTensor(['data', 'seg'], cast_to=None)
    tr_transforms = [SpatialTransform((544, 960), [272, 480], True,
                                      (0, 200), (11, 17), True, (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                                      angle_y=(0, 0), angle_z=(0, 0), do_scale=True, scale=(0.75, 1.25),
                                      border_mode_data='constant', order_seg=1, random_crop=True, p_el_per_sample=0.2,
                                      p_scale_per_sample=0.2, p_rot_per_sample=0.2)]

    tr_transforms.append(MirrorTransform(axes=(0, 1)))
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.1, per_channel=False))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.1, per_channel=False))
    tr_transforms.append(GammaTransform((0.6, 1.75), False, per_channel=False, retain_stats=True, p_per_sample=0.2))
    tr_transforms.append(numpy_to_tensor)
    tr_transforms = Compose(tr_transforms)
    transform = tr_transforms

    train_dir, val_file = load_file_dir()
    liver_dataset = LoadDataset(train_dir, transform)
    val_dataset = LoadDatasetVal(val_file)

    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=new_collate)  # drop_last=True
    print(len(liver_dataset))
    val_load = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    train_model(model, criterion, optimizer, dataloaders, val_load)


def train_model(model, criterion, optimizer, dataload, val_load, num_epochs=800):
    loss_list = []
    dice_list = []
    logs_dir = 'Logs/T{}/'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(logs_dir)
    writer = SummaryWriter(logs_dir)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size / args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss = []
        step = 0

        for batch in dataload:
            step += 1
            imgs = batch['data']
            imgs = imgs.to(torch.float32)
            inputs = imgs.cuda(device_ids[0])

            true_masks = batch['seg']
            true_masks = true_masks.to(torch.float32)
            labels = true_masks.cuda(device_ids[0])
            labels_quarter = F.max_pool2d(labels, kernel_size=(4, 4), stride=4, padding=0)
            labels_quarter = labels_quarter.type(torch.int64)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels_quarter)
            loss.backward()
            optimizer.step()
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))

            '''
            writer.add_images('images', inputs, epoch)
            writer.add_images('masks/true', labels_quarter, epoch)
            outputs_exp = torch.exp(outputs)
            output_max = outputs_exp.argmax(dim=1)
            output_max = output_max[:, None, :, :]
            output_max = output_max != 0
            writer.add_images('masks/pred', output_max, epoch)
            '''

        loss_list.append(epoch_loss_mean)
        tq.close()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss_mean))
        dice, iou = val_multi(model, criterion, val_load, new_num_classes, args.batch_size, device_ids)
        writer.add_scalar('Loss', epoch_loss_mean, epoch)
        writer.add_scalar('Dice', dice, epoch)
        writer.add_scalar('IoU', iou, epoch)
        dice_list.append([dice, iou])
        adjust_learning_rate(optimizer, epoch)
        torch.save(model.state_dict(), logs_dir + 'weights_{}.pth'.format(epoch))
        fileObject = open(logs_dir + 'LossList.txt', 'w')
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'dice_list.txt', 'w')
        for ip in dice_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()

    writer.close()
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=16)
    args = parse.parse_args()
    train()

    '''
    # A demo shows how the new_collate func works
    def cat_data_demo(catdata):
        return torch.cat(catdata, 0, out=None)


    def new_collate_demo(batch):
        # If data in batch is in dict
        if isinstance(batch[0], container_abcs.Mapping):
            #dict_a = {key: cat_data_demo([d[key] for d in batch]) for key in batch[0]}
            #print(dict_a['data'].shape)
            return {key: cat_data_demo([d[key] for d in batch]) for key in batch[0]}


    numpy_to_tensor = NumpyToTensor(['data', 'seg'], cast_to=None)
    tr_transforms = []
    tr_transforms.append(numpy_to_tensor)
    tr_transforms = Compose(tr_transforms)
    transform = tr_transforms

    train_dir, val_dir = load_file_dir()
    liver_dataset = LoadDataset(val_dir, transform)
    train_loader = DataLoader(liver_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, 
                                collate_fn=new_collate_demo)
    #examples = iter(train_loader)
    #example_data = examples.next()

    for batch in train_loader:
        imgs = batch['data']
        imgs = imgs.to(torch.float32)
        inputs = imgs.cuda(device_ids[0])
        print(inputs.shape)
    '''
