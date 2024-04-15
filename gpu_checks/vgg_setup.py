import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1
import cv2
import os

device = torch.device('cuda')
resnet = InceptionResnetV1(pretrained='casia-webface').to(device)
x = torch.randn(1, 3, 224, 224).to(device)
output = resnet(x)
print("Output shape of InceptionResnetV1:", output.shape)
