import numpy as np
import torch
import torchvision

from config import IMAGE_MEAN, IMAGE_STD, USE_CUDA, DATA_LOCATION


def __resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    return model, model.avgpool


def __densenet121():
    model = torchvision.models.densenet121(pretrained=True)
    return model, model.features


def __mnasnet1_0():
    model = torchvision.models.mnasnet1_0(pretrained=True)
    return model, model.layers


def create_model(model_name):
    if model_name == "resnet50":
        model, features = __resnet50()
    elif model_name == "densenet121":
        model, features = __densenet121()
    elif model_name == "mnasnet1.0":
        model, features = __mnasnet1_0()

    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    if USE_CUDA:
        model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    return torch.nn.Sequential(model, softmax), features


def create_image_saving_model(model_name, xai_name):
    input_image_saving_hook = InputImageSavingHook(xai_name)
    model, features = create_model(model_name)
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


class EncodingSavingHook:
    def __init__(self, model_name, image_class, xai_name):
        self.encoding_store = []
        self.counter = torch.zeros(1, dtype=torch.int32)
        self.xai_name = xai_name
        self.model_name = model_name
        self.image_class = image_class

    def hook(self, module, input_, output):
        output = output.detach()
        if output.dim() > 2:
            output = output.mean([2, 3])
        self.encoding_store.append(output.cpu().numpy())
        if len(self.encoding_store) > 10000:
            self.save_encodings()

    def save_encodings(self):
        self.counter = self.counter.add(1)
        np.save(
            f"{DATA_LOCATION}/{self.model_name}_{self.image_class}_{self.xai_name}_{self.counter.numpy()[0]}.npy",
            np.concatenate(self.encoding_store, axis=0))
        self.encoding_store.clear()
