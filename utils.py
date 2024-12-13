import torch
import torchvision.models as models
from torchinfo import summary


def save_neural_model(model,
                      path_state_dict="resnet18_noisesmog_dict_2024.pth"):
    # saving model
    net_state_dict = model.state_dict()
    torch.save(net_state_dict, path_state_dict)
    print("model state saving")


def load_neural_model(model,
                      path_state_dict="resnet18_noisesmog_dict_2024.pth"):
    model_state_dict = torch.load(path_state_dict)
    model.load_state_dict(model_state_dict)
    return model