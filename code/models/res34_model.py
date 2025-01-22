import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet34', pretrained=True)
resnet50.to(device)

print(summary(resnet50, (3, 160, 40)))