import torch
import torchvision

from config import MODEL_NAME, IMAGE_MEAN, IMAGE_STD


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

    return model, features


def create_image_saving_model(xai_name):
    input_image_saving_hook = InputImageSavingHook(xai_name)
    model, features = create_model()
    model.register_forward_hook(input_image_saving_hook.hook)
    return model


class InputImageSavingHook:
    def __init__(self, xai_model_name):
        self.counter = 0
        self.xai_model_name = xai_model_name

    def hook(self, module, input_, output):
        tensor_image_std = torch.as_tensor(IMAGE_STD).view(-1, 1, 1)
        tensor_image_mean = torch.as_tensor(IMAGE_MEAN).view(-1, 1, 1)
        for i in range(0, len(input_)):
            self.counter += 1
            unnormalized_input = input_[i].mul(tensor_image_std).add(tensor_image_mean)
            torchvision.utils.save_image(unnormalized_input[i], f"{self.xai_model_name}.{self.counter}.png")
