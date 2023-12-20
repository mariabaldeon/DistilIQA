import torch
from einops import rearrange
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ClassModel(nn.Module): 
    """
    Defines the structure of the Transformer model
    in_channels: number of channels in input image
    model: Transformer model to be located after the convolutional stem
    n_features: number of features for the first convolutional block
    n_conv_stems: number of convolutional blocks in the convolutional stem
    """
    def __init__(self, in_channels, model, n_features, n_conv_stems):
        super(ClassModel, self).__init__() 
        self.n_conv_stems=n_conv_stems
        if self.n_conv_stems>0:             
            self.conv_head= nn.Sequential(self.conv_block(in_channels, n_features,3, 1),
                *[self.conv_block((2**i)*n_features, (2**(i+1))*n_features,3, 1) for i in range(n_conv_stems-1)])
            self.downsample= self.conv_block((2**(n_conv_stems-1))*n_features, 3, 2, 2)
        else: 
            self.downsample= self.conv_block(in_channels, 3, 2, 2)            
        self.model= model
    def conv_block(self, in_channels, out_channels,kernel_size, stride): 
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
    def forward(self, x): 
        if self.n_conv_stems>0: 
            x = self.conv_head(x)
            #print(x.shape)
        x = self.downsample(x)
        #print(x.shape)
        x = x[:,:,16:240, 16:240]
        x = self.model(x)
        return x

class DefineModel():
    """
    Defines the Transformer model 
    model: Transformer model to be located after the convolutional stem, we use the ViT base architecture
    vit: Bool, True: if the architecture after the convolutional stem is ViT False: if you use other architecture
    classes: number of values for prediction. Teacher members use 1, student use 5 as it predicts a vector with 5 components
    in_channels: number of channels in input image
    n_features: number of features for the first convolutional block
    n_conv_stems: number of convolutional blocks in the convolutional stem
    """
    def __init__(self, model, vit: bool, classes: int, in_channels: int=1, n_features: int=8, n_conv_stems: int=5): 
        self.model = model
        self.classes=classes
        self.in_channels=in_channels
        self.vit=vit
        self.n_features=n_features
        self.n_conv_stems=n_conv_stems
    
    def set_model(self): 
        if self.vit:
            self.model.heads=nn.Linear(in_features=self.model.heads.head.in_features, out_features=self.classes, bias=True)
        else:
            self.model.fc=nn.Linear(in_features=self.model.fc.in_features, out_features=self.classes, bias=True)
        return ClassModel(self.in_channels, self.model, self.n_features, self.n_conv_stems)


if __name__=="__main__":
    # Initialize ViT base architecture
    vit=vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # Add convolutional stem
    model = DefineModel(model=vit, vit=True,classes=1, in_channels=1, n_features=8, n_conv_stems=5).set_model()
    print(model)
    input = torch.randn(1, 1, 512, 512)
    output = model(input)
    print(output.shape)




