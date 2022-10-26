import torch
import torch.nn as nn
import torchvision

import src.model.submodules.classify_head as classify_head


class JHTModel(nn.Module):

    def __init__(self, model_config) -> None:
        super().__init__()
        model = {
            "resnet18": torchvision.models.resnet18(pretrained=True),
            "resnet34": torchvision.models.resnet34(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
            "vgg11": torchvision.models.vgg11(pretrained=True),
            "vgg19": torchvision.models.vgg19(pretrained=True),
            "inception_v3": torchvision.models.inception_v3(pretrained=True, aux_logits=False),
            "densenet121": torchvision.models.densenet121(pretrained=True),
            "shufflenetv2": torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        }[model_config["pretrained_arch"]]
        print(model)
        # self.image_head = model.features
        self.image_head = nn.Sequential(*list(model.children())[:-1]) # 0: -1 for resnet
        # self.image_head = model
        # self.image_head.dropout = nn.Identity()
        # self.image_head.fc = nn.Identity()

        self.classifier = classify_head.get_instance(
            model_config["classifier_type"], model_config["classifier_params"])

    def forward(self, input_dict):
        prob_dict = {}
        # print(input_dict["image_data"].shape)
        image_embedding= self.image_head(input_dict["image_data"])
        image_embedding = image_embedding.view(input_dict["image_data"].size(0), -1).squeeze()
        prob_dict["output"] = self.classifier(image_embedding)
        return prob_dict["output"]


# get_img_feature_by_vit()
# for value in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
#     get_img_feature_by_pretrained_model(
#         image_dir=IMAGE_DIR,
#         # data_type='sequence',
#         arch=value
#     )
