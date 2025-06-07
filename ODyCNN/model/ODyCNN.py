import torch.nn as nn
import torch
from model.ODyConv import ODyConv2d
from model.MSDilateFormer import *
from model.softpool import *
from fvcore.nn import FlopCountAnalysis

def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)
        
    def conv_block(self, input_channels, k):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.LeakyReLU(),
            ODyConv2d(input_channels, k, kernel_size=3, stride=1, padding=1))  
            
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X
    
class Downsample_layer(nn.Module):
    def __init__(self):
        super(Downsample_layer,self).__init__()
        self.downsample = SoftPool2d(kernel_size=2)             
    def forward(self, x):             
        out = self.downsample(x)               
        return out

class   Upsample_layer(nn.Module):
    def __init__(self, in_channel):
        super(Upsample_layer,self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel), nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)) 
        
    def forward(self, x):             
        x1 = self.conv(x)      
        return x1
    
class Bottleneck(nn.Module):
    def __init__(self, in_channel, k):
        super(Bottleneck,self).__init__()

        self.Trans_path =  MSDilateformer(
                            H=2, W=128, Ph=1, Pw=4, in_chans=256, 
                            embed_dim=96, hidden_dim=16,
                            depths=[3], num_heads=[3])
        self.Conv_path = nn.Sequential(DenseBlock(in_channel=in_channel, k=k), nn.Conv2d(4*k+ in_channel, in_channel, 1))
        self.fuse = nn.Sequential(nn.Conv2d(in_channel*2, in_channel, kernel_size=1))
    
    def forward(self, x):
        x1 = self.Trans_path (x)
        x_trans = x1 + x
        x2 = self.Conv_path(x)
        x_conv = x2 + x
        x3 = torch.cat((x_trans, x_conv), dim=1)      
        x4 = self.fuse(x3)
        return x4
    
class skip(nn.Module):
    def __init__(self, in_channel):
        super(skip,self).__init__()
        self.connc = nn.Sequential(
            nn.BatchNorm2d(in_channel), nn.LeakyReLU(),
            nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, padding=0))   
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim = 1)
        out = self.connc(x)  
        return out

class ODyCNN(nn.Module):
    def __init__(self):
        super(ODyCNN, self).__init__()

        ##########    Preliminary feature extraction    ############
        self.shallow_feature = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding = 1),
                                        nn.BatchNorm2d(32), nn.LeakyReLU())    

        ##########   Downsampling   ############
        self.Dense_down1 = DenseBlock(in_channel=32, k=8)
        self.Downsample1 = Downsample_layer()
        self.Dense_down2 = DenseBlock(in_channel=64, k=16)
        self.Downsample2 = Downsample_layer()
        self.Dense_down3 = DenseBlock(in_channel=128, k=32)     
        self.Downsample3 = Downsample_layer()
        
        ##########   Bottleneck   ############
        self.bottleneck = Bottleneck(in_channel=256, k=64)
        
        ##########   upsampling   ############
        self.Upsample3 = Upsample_layer(in_channel=256)
        self.skip3 = skip(512)
        self.Dense_up3 = nn.Sequential(DenseBlock(in_channel=128, k=32), nn.Conv2d(256, 128, 1, 1, 0))
        self.Upsample2 = Upsample_layer(in_channel=128)
        self.skip2 = skip(256)
        self.Dense_up2 = nn.Sequential(DenseBlock(in_channel=64, k=16), nn.Conv2d(128, 64, 1, 1, 0))
        self.Upsample1 = Upsample_layer(in_channel=64)
        self.skip1 = skip(128)
        self.Dense_up1 = nn.Sequential(DenseBlock(in_channel=32, k=8), nn.Conv2d(64, 32, 1, 1, 0))
        ##########    Out layer    ############
        self.out_layer = nn.Sequential(nn.BatchNorm2d(32), nn.LeakyReLU(), nn.Conv2d(32, 1, 3, 1, 1)) 

    def forward(self, x):
        ##  input_size[64, 1, 26, 1024] 
        x1 = self.shallow_feature(x)       ##  [64, 32, 16, 1024]
        x2 = self.Dense_down1(x1)          ##  [64, 64, 16, 1024]
        x3 = self.Downsample1(x2)          ##  [64, 64, 8, 512]
        x4 = self.Dense_down2(x3)          ##  [64, 128, 8, 512]
        x5 = self.Downsample2(x4)          ##  [64, 128, 4, 256]
        x6 = self.Dense_down3(x5)          ##  [64, 256, 4, 256]
        x7 = self.Downsample3(x6)          ##  [64, 256, 2, 128]
        x8 = self.bottleneck(x7)           ##  [64, 256, 2, 128]
        x9 = self.Upsample3(x8)            ##  [64, 256, 4, 256]
        x10 = self.skip3(x9, x6)           ##  [64, 128, 4, 256]
        x11 = self.Dense_up3(x10)          ##  [64, 128, 4, 256]
        x12 = self.Upsample2(x11)          ##  [64, 128, 8, 512]
        x13 = self.skip2(x12, x4)          ##  [64, 64, 8, 512]
        x14 = self.Dense_up2(x13)          ##  [64, 64, 8, 512]
        x15 = self.Upsample1(x14)          ##  [64, 64, 16, 1024]
        x16 = self.skip1(x15, x2)          ##  [64, 32, 16, 1024]
        x17 = self.Dense_up1(x16)          ##  [64, 32, 16, 1024]
        out = self.out_layer(x17)          ##  [64, 1, 16, 1024]

        mask = torch.zeros_like(x)
        mask[x == 0] = 1
        output = torch.mul(mask, out) + x
           
        return output


if __name__ == "__main__":
    x = torch.rand([16, 1, 16, 1024])
    net = ODyCNN()
    y = net(x)
    print(y.shape)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    flops = FlopCountAnalysis(net, torch.rand(16, 1, 16, 1024))
    print('Params (M):', n_parameters/1000000)     
    print('Flops (G):', flops.total()/1000000000)  