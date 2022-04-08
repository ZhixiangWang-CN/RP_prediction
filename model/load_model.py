import torch
from torch import nn
from torch.nn.functional import *
from model import resnet




class Classification_Net(nn.Module):
    def __init__(self, model):
        super(Classification_Net, self).__init__()

        self.w = nn.Parameter(torch.ones(2))


        self.resnet = model

        self.classififcation = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 1, bias=True),
        )



    def forward(self, x):

        CT = x[:, 0:1, :, :, :]
        RD = x[:, 1:, :, :, :]

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        x = CT * w1 + RD * w2
        x = self.resnet(x)
        x = self.classififcation(x)
        return x



def load_model(structure='Classification'):

    res_model = resnet.generate_model(model_depth=10,
                                          n_input_channels=1,n_classes = 1)

    if structure=='Classification':
        model = Classification_Net(res_model)

    return model