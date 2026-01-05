
import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes, model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
         model = models.resnet50(pretrained=pretrained)
         num_ftrs = model.fc.in_features
         model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model
