import numpy as np
import torch
import torchvision

from config import MODEL_NAME, IMAGE_MEAN, IMAGE_STD, USE_CUDA, IMAGE_CLASS


def __resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    return model, model.avgpool


def __densenet121():
    model = torchvision.models.densenet121(pretrained=True)
    return model, model.features

def __mnasnet1_0():
    model = torchvision.models.mnasnet1_0(pretrained=True)
    return model, model.layers


def create_model():
    if MODEL_NAME == "resnet50":
        model, features = __resnet50()
    elif MODEL_NAME == "densenet121":
        model, features = __densenet121()
    elif MODEL_NAME == "mnasnet1.0":
        model, features = __mnasnet1_0()

    else:
        raise NotImplementedError(f"{MODEL_NAME} is not implemented")
    if USE_CUDA:
        model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    return torch.nn.Sequential(model, softmax), features


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


class EncodingSavingHook:
    def __init__(self, name):
        self.encoding_store = []
        self.name = name
        self.counter = 0

    def hook(self, module, input_, output):
        out = output.clone()
        if out.dim() > 2:
            out = out.mean([2, 3])
        self.encoding_store.append(out.detach().cpu().numpy().copy())
        # if self.encoding_store is None:
        #     self.encoding_store = encodings
        # else:
        #     self.encoding_store = np.concatenate([self.encoding_store, encodings], axis=0)
        # self.counter += 1
        #
        # if (self.counter % 1000) == 0:
        #     self.save_encodings()

    def save_encodings(self):
        np.save(f"{MODEL_NAME}_{IMAGE_CLASS}_{self.name}.npy", np.concatenate(self.encoding_store, axis=0))
