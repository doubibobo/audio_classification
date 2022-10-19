from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
import torch
import torch.nn as nn

model_arch = {
    'resnet18': {
        'block': BasicBlock,
        'layers': [2, 2, 2, 2]
    },
    'resnet34': {
        'block': BasicBlock,
        'layers': [3, 4, 6, 3]
    },
    'resnet50': {
        'block': Bottleneck,
        'layers': [3, 4, 6, 3],
    },
    'resnet101': {
        'block': Bottleneck,
        'layers': [3, 4, 23, 3]
    },
    'resnet151': {
        'block': Bottleneck,
        'layers': [3, 8, 36, 3]
    }
}


class ResNet(nn.Module):
    def __init__(self, model_name='resnet18'):
        super().__init__()
        resnet_model = _resnet(
            model_name,
            model_arch[model_name]['block'],
            model_arch[model_name]['layers'],
            pretrained=True,
            progress=True
        )
        feature_layers = list(resnet_model.children())[0: -1]
        self.model = nn.Sequential(*feature_layers)

    def forward(self, input_image):
        feature_output = self.model(input_image)
        output = torch.flatten(feature_output, 1)
        return output


if __name__ == '__main__':
    model = ResNet()
    input_image = torch.randn((1, 3, 224, 224))
    output = model(input_image)
    print(output.shape)
