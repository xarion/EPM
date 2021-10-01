tennis_ball_class = 852
printer_class = 742
chocolate_sauce_class = 960

MODEL_NAMES = ["mnasnet1.0", "densenet121", "resnet50"]
IMAGE_CLASSES = [tennis_ball_class, printer_class, chocolate_sauce_class]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

USE_CUDA = True

DATA_LOCATION = '/disks/bigger/xai_methods/224'