import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
from IPython.core.debugger import set_trace
import torchvision.models as models

# Weights initializer 
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Pretrain VGG19
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Extracting features at different level
        self.feature_extractor1 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.feature_extractor2 = nn.Sequential(*list(vgg19_model.features.children())[:17])
        self.feature_extractor3 = nn.Sequential(*list(vgg19_model.features.children())[:8])
        
    def forward(self, img):
        out1 = self.feature_extractor1(img)
        return out1
# Normalization layer
# This normalization part is borrowed from https://nvlabs.github.io/SPADE/
class NLayer_torch(nn.Module):
    def __init__(self,ch):
        super(NLayer_torch, self).__init__()
        self.NL_scale_conv0 = nn.Conv2d(3, ch//2, 3,1,1)
        self.NL_scale_conv1 = nn.Conv2d(ch//2, ch, 3,1,1)
        self.NL_shift_conv0 = nn.Conv2d(3, ch//2, 3,1,1)
        self.NL_shift_conv1 = nn.Conv2d(ch//2, ch, 3,1,1)
        self.norm = nn.BatchNorm2d(ch, affine=True)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        Norm = self.norm(x[0])
        scale = self.NL_scale_conv1(F.relu(self.NL_scale_conv0(x[1]), inplace=True))
        shift = self.NL_shift_conv1(F.relu(self.NL_shift_conv0(x[1]), inplace=True))
        return Norm * (scale + 1) + shift 
    

# Residual block   
class ResBl(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBl, self).__init__()
        
        self.conv1 = nn.Conv2d(in_size, (in_size + out_size)//2, 3, 1, 1, bias=True)
        self.NL1 = NLayer_torch((in_size))
        
        self.conv2 = nn.Conv2d((in_size + out_size)//2, (in_size + out_size)//2, 3, 1, 1, bias=True)
        self.NL2 = NLayer_torch((in_size + out_size)//2)
        
        self.conv3 = nn.Conv2d((in_size + out_size)//2, out_size, 3, 1, 1, bias=True)
        self.NL3 = NLayer_torch((in_size + out_size)//2)
        

    def forward(self, x):
        out1 = F.relu(self.NL1((x[0],x[1])))
        out2 = self.conv1(out1)
        out3 = F.relu(self.NL2((out2,x[1])))
        out4 = self.conv2(out3)
        out5 =  F.relu(self.NL3((out4,x[1])))
        out6 = self.conv3(out5)
        
        return out6 

# Concatenation block    
class UpFeat(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpFeat, self).__init__()
        
        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True)
        self.NL1 = NLayer_torch(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1, bias=True)
    def forward(self, x):
        
        
        out1 = self.conv1(x[0])
        out2 = F.relu(self.NL1((out1,x[1])))
        out3 = self.conv2(out2)
        
        return out3
    
class MainBl(nn.Module):
    def __init__(self, in_size, out_size):
        super(MainBl, self).__init__()
        
        self.resbl = ResBl(in_size,out_size)
        
        
        self.conv = nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True)
        self.NL = NLayer_torch(in_size)
    def forward(self, x):
        out1 = self.resbl(x)
        out2 =  F.relu(self.NL((x[0],x[1])))
        out3 = self.conv(out2)
        return out1 + out3

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        
        
        self.ch1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=True))
        self.ch2 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=True))
        self.ch3 = nn.Sequential(nn.Conv2d(1, 128, 3, 1, 1, bias=True))
        self.ch4 = nn.Sequential(nn.Conv2d(1, 256, 3, 1, 1, bias=True))
        
        
        self.ch5 = UpFeat(512,256)
        self.ch6 = UpFeat(256,128)
        self.ch7 = UpFeat(128,64)


        
        self.down1 = MainBl(32, 64)
        self.down2 = MainBl(64, 128)
        self.down3 = MainBl(128, 256)
        self.down4 = MainBl(256, 256)
        
        self.up1 = MainBl(256, 128)
        self.up2 = MainBl(128, 64)
        self.up3 = MainBl(64, 32)
        
        self.conv = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1, bias=True),nn.Tanh())
        
    

    def forward(self, x):

        # U-Net generator with skip connections from encoder to decoder
        ###############   LEVEL 3
        d1 = self.ch1(x[0])  
        d2 = self.down1((d1,x[1]))
        d2_ = self.pooling(d2)  
        x_1 = self.pooling(x[0])  
        c_1 = self.pooling(x[1]) 
        
        d3 = self.ch2(x_1)
        d4 = self.down2((d3+d2_,c_1))
        d4_ = self.pooling(d4)  
        x_2 = self.pooling(x_1) 
        c_2 = self.pooling(c_1) 
        
        d5 = self.ch3(x_2)
        d6 = self.down3((d5+d4_,c_2))
        d6_ = self.pooling(d6)  
        x_3 = self.pooling(x_2) 
        c_3 = self.pooling(c_2) 
        
        d7 = self.ch4(x_3)
        d8 = self.down4((d7+d6_,c_3))
        u1 = self.upsampling(d8)
        M1 = torch.cat((d6, u1), dim=1)
     
        d9 = self.ch5((M1,c_2))
        d10 = self.up1((d9,c_2))
        u2 = self.upsampling(d10)
        M2 = torch.cat((d4, u2), dim=1)
        
        d11 = self.ch6((M2,c_1))
        d12 = self.up2((d11,c_1))
        u3 = self.upsampling(d12)
        M3 = torch.cat((d2, u3), dim=1)      
        
        d13 = self.ch7((M3,x[1]))
        d14 = self.up3((d13,x[1]))
        

        
        uf = self.conv(d14)
#
        
     
        return uf

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pooling = nn.AvgPool2d(6)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True),
            
            

        )

         
        self.gan1 = nn.Sequential(
            nn.Linear(512*1*1, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1)
        )





    def forward(self, x):

        fea = self.feature(x[:,0:1,:,:])
        fea = self.pooling(fea)
        fea = fea.view(fea.size(0), -1)
        gan1 = self.gan1(fea)
        return gan1  
    