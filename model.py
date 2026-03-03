import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()

        # --- Encoder (Downward Path) ---
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = self._conv_block(64, 128)

        # --- Attention Gates ---
        # F_g: channels from the decoder (gating signal)
        # F_l: channels from the encoder (skip connection)
        # F_int: intermediate channels for the internal alignment

        self.att2 = AttentionGate3D(F_g=64, F_l=64, F_int=64)  # Changed F_g from 128 to 64
        self.att1 = AttentionGate3D(F_g=32, F_l=32, F_int=32)  # Changed F_g from 64 to 32
        
        # --- Decoder (Upward Path) ---
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64) 
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        # --- Output Layer ---
        self.final = nn.Sequential(
            nn.Conv3d(32, out_channels, kernel_size=1),
            nn.Sigmoid() 
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)              # High res, low features (32, 32, 32, 32)
        e2 = self.enc2(self.pool(e1))  # Mid res, mid features (64, 16, 16, 16)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e2)) # Low res, deep features (128, 8, 8, 8)

        # Decoder Step 1
        d2 = self.up2(b)               # Up to (64, 16, 16, 16)
        
        # ATTENTION GATE 2: Use d2 (the "boss") to filter e2 (the skip connection)
        e2_att = self.att2(g=d2, x=e2)
        
        d2 = torch.cat([d2, e2_att], dim=1) 
        d2 = self.dec2(d2)

        # Decoder Step 2
        d1 = self.up1(d2)              # Up to (32, 32, 32, 32)
        
        # ATTENTION GATE 1: Use d1 to filter e1
        e1_att = self.att1(g=d1, x=e1)
        
        d1 = torch.cat([d1, e1_att], dim=1) 
        d1 = self.dec1(d1)
        
        return self.final(d1)

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)