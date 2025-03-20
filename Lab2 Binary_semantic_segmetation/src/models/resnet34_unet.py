# # Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return self.relu(out)

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNetEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        shortcut = None
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, shortcut)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.encoder(x)      # 256*256*3 -> 64*64*64
        x2 = self.layer1(x1)      # 64*64*64 -> 64*64*64
        x3 = self.layer2(x2)      # 64*64*64 -> 32*32*128
        x4 = self.layer3(x3)      # 32*32*128 -> 16*16*256
        x5 = self.layer4(x4)      # 16*16*256 -> 8*8*512
        return x1, x2, x3, x4, x5
    
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        # resnet output size is different from unet, so we need to interpolate
        if skip_connection is not None:
            if x.shape[2:] != skip_connection.shape[2:]:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_connection], dim=1)
        return self.decoder(x)

class ResNet34_Unet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(ResNet34_Unet, self).__init__()
        
        # Encoder
        self.encoder = ResNetEncoder(in_channels)
        
        # Decoder
        self.decoder1 = UNetDecoder(512, 256, 256) # 8*8*512 -> 16*16*256
        self.decoder2 = UNetDecoder(256, 128, 128) # 16*16*256 -> 32*32*128
        self.decoder3 = UNetDecoder(128, 64, 64) # 32*32*128 -> 64*64*64
        self.decoder4 = UNetDecoder(64, 64, 64) # 64*64*64 -> 128*128*64
        self.decoder5 = UNetDecoder(64, 0, 64) # 128*128*64 -> 256*256*64
        
        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1) # 256*256*64 -> 256*256*1
        
    def forward(self, x):
        # x1: 64*64*64, x2: 64*64*64, x3: 32*32*128, x4: 16*16*256, x5: 8*8*512
        x1, x2, x3, x4, x5 = self.encoder(x) 
        
        x = self.decoder1(x5, x4) # 8*8*512 -> 16*16*256
        x = self.decoder2(x, x3) # 16*16*256 -> 32*32*128
        x = self.decoder3(x, x2) # 32*32*128 -> 64*64*64
        x = self.decoder4(x, x1) # 64*64*64 -> 128*128*64
        x = self.decoder5(x, None) # 128*128*64 -> 256*256*64
        
        x = self.output(x) # 256*256*64 -> 256*256*1
    
        return x
    

# # 測試 ResNet34-UNet
# if __name__ == "__main__":
#     model = ResNet34_Unet(in_channels=3, out_channels=1)
#     x = torch.randn(1, 3, 256, 256)  # 測試輸入
#     y = model(x)
#     print(f"輸出形狀: {y.shape}")  # 預期: torch.Size([1, 1, 256, 256])
