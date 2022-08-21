import torch
from LWANet import AFB
import LWANet
import torch.nn as nn


mod = LWANet.LWANet(num_classes=5, pretrained=True)
weight_load = 'Logs/T20211209_103331/weights_757.pth'
# Loading model weight
mod.load_state_dict(
    {k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=torch.device("cpu")).items()})

for param in mod.parameters():
    param.requires_grad = False

mod.afb1 = AFB(24, 24)

print('Modified layer requires_grad is True')
for name, param, in mod.named_parameters():
    if param.requires_grad:
        print(name)