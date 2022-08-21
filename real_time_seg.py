import torch
from LWANet import LWANet
from torch import nn
import cv2
import numpy as np
import torchvision.transforms as transforms


weight_load = 'Logs/T20200911_211620/weights_659.pth'
device = torch.device("cuda")
model = LWANet(num_classes=8, pretrained=True)
# Check model structure
# print(model)

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

#cap = cv2.VideoCapture('dataset/WIN_20211208_14_40_05_Pro.mp4')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

with torch.no_grad():
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (960, 544))
        cv2.imshow('frame', image)
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.cuda(0)
        outputs = model(input_batch)

        outputs_exp = torch.exp(outputs)
        output_max = outputs_exp.argmax(dim=1)
        output_max = output_max.cpu().numpy()
        output_max = np.squeeze(output_max, axis=0)
        output_max = output_max.astype('float32')
        vis_output = cv2.cvtColor(output_max, cv2.COLOR_GRAY2BGR)
        cv2.imshow('output', vis_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
