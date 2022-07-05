import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, out_size=1):
        super(ResNet, self).__init__()
        # self.device = torch.device('cuda')
        self.resnet = models.resnet34(pretrained=True)
        ### fixed feature extractor
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_size)
        #self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.resnet.fc.in_features, out_size))
        self.resnet.fc = self.resnet.fc.requires_grad_(True)

    def forward(self, image):
        # image = image.to(self.device)
        pred = self.resnet(image)
        pred = pred.view(pred.shape[0], )  # reshape from (N,1) to (N,) to avoid mismatches in the loss function
        # return self.sigmoid(self.dropout(self.relu(features))).squeeze(1)
        return pred