import torch
import torchvision
from captum.attr._utils.lrp_rules import IdentityRule
from torchvision.transforms import transforms

from config import MODEL_NAME


def __resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    return model, model.avgpool


def __densenet121():
    model = torchvision.models.densenet121(pretrained=True)
    return model, model.features


def create_model():
    if MODEL_NAME == "resnet50":
        model, features = __resnet50()
    elif MODEL_NAME == "densenet121":
        model, features = __densenet121()
    else:
        raise NotImplementedError(f"{MODEL_NAME} is not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return torch.nn.Sequential(normalize, model), features


def create_image_saving_model(xai_name):
    input_image_saver = InputImageSavingModule(xai_name)
    model, features = create_model()
    return torch.nn.Sequential(input_image_saver, model)


class InputImageSavingModule(torch.nn.Module):
    def __init__(self, xai_model_name):
        super(InputImageSavingModule, self).__init__()
        self.counter = 0
        self.xai_model_name = xai_model_name
        self.rule = IdentityRule()

    def forward(self, x):
        with torch.no_grad():
            for i in range(0, x.shape[0]):
                self.counter += 1
                torchvision.utils.save_image(x[i], f"{self.xai_model_name}.{self.counter}.png")
        return x
