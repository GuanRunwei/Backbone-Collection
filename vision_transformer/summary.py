import torchsummary
from vision_transformer.vit import vit


if __name__ == '__main__':
    print(torchsummary.summary(vit(pretrained=True), input_size=(3, 224, 224)))