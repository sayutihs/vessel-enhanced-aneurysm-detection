import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3d(nn.Module):
    """(Conv3d -> BatchNorm3d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down3d(nn.Module):
    """Downscaling with MaxPool3d then DoubleConv3d"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3d(nn.Module):
    """Upscaling then DoubleConv3d"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3d(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # In case of size mismatch due to division rounding
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VesselEnhancementSubnet(nn.Module):
    """
    Lightweight 3D U-Net designed to enhance vascular structures
    and output a single-channel vessel probability/enhancement map.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Small-scale 3D UNet for speed
        self.inc = DoubleConv3d(in_channels, 16)
        self.down1 = Down3d(16, 32)
        self.down2 = Down3d(32, 64)
        
        self.up1 = Up3d(96, 32)  # 64 + 32 channels concat
        self.up2 = Up3d(48, 16)  # 32 + 16 channels concat
        self.outc = OutConv3d(16, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        vesselness = torch.sigmoid(self.outc(x))
        return vesselness


class AneurysmDetectionSubnet(nn.Module):
    """
    A 3D U-Net designed to segment aneurysms.
    It takes the concatenated raw volume and vessel-enhanced map.
    """
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        # Main segmentation UNet
        self.inc = DoubleConv3d(in_channels, 32)
        self.down1 = Down3d(32, 64)
        self.down2 = Down3d(64, 128)
        self.down3 = Down3d(128, 256)
        
        self.up1 = Up3d(384, 128)  # 256 + 128 channels concat
        self.up2 = Up3d(192, 64)   # 128 + 64 channels concat
        self.up3 = Up3d(96, 32)    # 64 + 32 channels concat
        self.outc = OutConv3d(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        aneurysm_logits = self.outc(x)
        return aneurysm_logits


class VesselEnhancedAneurysmNet(nn.Module):
    """
    Complete end-to-end Vessel-Enhanced Deep Learning Model.
    1. Feeds raw input to Vessel Enhancement Subnet.
    2. Concatenates raw input and the generated Vesselness Map.
    3. Feeds combined features to Aneurysm Detection Subnet.
    """
    def __init__(self):
        super().__init__()
        self.vessel_subnet = VesselEnhancementSubnet(in_channels=1, out_channels=1)
        self.aneurysm_subnet = AneurysmDetectionSubnet(in_channels=2, out_channels=1)

    def forward(self, x):
        # x is raw volume of shape [B, 1, D, H, W]
        vesselness = self.vessel_subnet(x)
        
        # Concatenate raw image and vesselness map along the channel dimension
        # x_concat shape: [B, 2, D, H, W]
        x_concat = torch.cat([x, vesselness], dim=1)
        
        # Output aneurysm logits (we apply sigmoid during training loss or inference)
        aneurysm_logits = self.aneurysm_subnet(x_concat)
        
        return aneurysm_logits, vesselness

# Sanity check if run directly
if __name__ == "__main__":
    model = VesselEnhancedAneurysmNet()
    # Dummy input representing batch=2, channel=1, size 32x32x32 to run fast on CPU
    dummy_input = torch.randn(2, 1, 32, 32, 32)
    logits, vesselness = model(dummy_input)
    print("Model verification successful:")
    print(f"  Input shape:      {dummy_input.shape}")
    print(f"  Vesselness shape: {vesselness.shape} (min: {vesselness.min().item():.3f}, max: {vesselness.max().item():.3f})")
    print(f"  Aneurysm logits:  {logits.shape}")
