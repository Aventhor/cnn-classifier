import torch

from torchvision.datasets import ImageFolder

from .cnn_model import CnnModel
from utils.device import to_device


def predict_image(dataset: ImageFolder, img, model: CnnModel, device: torch.device):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]