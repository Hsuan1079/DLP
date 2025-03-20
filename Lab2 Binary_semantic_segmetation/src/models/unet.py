# Implement your UNet model here
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x) 
        x = torch.cat([x, skip_connection], dim=1) 
        return self.decoder(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = Encoder(in_channels, 64)  # 256*256*3 -> 256*256*64
        self.encoder2 = Encoder(64, 128)  # 256*256*64 -> 128*128*128
        self.encoder3 = Encoder(128, 256) # 128*128*128 -> 64*64*256
        self.encoder4 = Encoder(256, 512) # 64*64*256 -> 32*32*512
        self.encoder5 = Encoder(512, 1024) # bottleneck 32*32*512 -> 16*16*1024
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Decoder
        self.decoder1 = Decoder(1024, 512) # 16*16*1024 -> 32*32*512
        self.decoder2 = Decoder(512, 256) # 32*32*512 -> 64*64*256
        self.decoder3 = Decoder(256, 128) # 64*64*256 -> 128*128*128
        self.decoder4 = Decoder(128, 64) # 128*128*128 -> 256*256*64
        
        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1) # 256*256*64 -> 256*256*1 

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x) # 256*256*3 -> 256*256*64
        x2 = self.encoder2(self.pool(x1)) # self.pool(x1) -> 128*128*64 endcode -> 128*128*128
        x3 = self.encoder3(self.pool(x2)) # 64*64*128 -> 64*64*256
        x4 = self.encoder4(self.pool(x3)) # 32*32*256 -> 32*32*512
        x5 = self.encoder5(self.pool(x4)) # bottleneck 16*16*512 -> 16*16*1024
        
        # Decoder
        x = self.decoder1(x5, x4) # 16*16*1024 -> 32*32*512
        x = self.decoder2(x, x3) # 32*32*512 -> 64*64*256
        x = self.decoder3(x, x2) # 64*64*256 -> 128*128*128
        x = self.decoder4(x, x1) # 128*128*128 -> 256*256*64
        
        # Output
        return self.output(x) # 256*256*64 -> 256*256*1