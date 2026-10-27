import torch
import torchvision.models as models
import torch.nn as nn
class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.fc_layers = nn.Sequential(
            nn.Dropout(p= 0.2),
            nn.Linear(2048, num_classes) #2048: resnet50.fc.in_features
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.fc_layers(x)
        return x


def build_model(num_classes: int):
    return DogBreedClassifier(num_classes= num_classes)
