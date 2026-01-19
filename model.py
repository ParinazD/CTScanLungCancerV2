import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """
    3D U-Net for Lung Nodule Segmentation.
    """
    def __init__(self, in_channels=1, out_channels=1):
        """
        @param in_channels: (int) 1 for grayscale CT scans.
        @param out_channels: (int) 1 for the binary probability mask.
        """
        super(UNet3D, self).__init__()

        # encoder - downward path
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        #
        self.bottleneck = self._conv_block(64, 128)

        # decoder - forward path 
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64) # 64 from up + 64 from skip connection
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32) # 32 from up + 32 from skip connection

        # output layer
        self.final = nn.Sequential(
            nn.Conv3d(32, out_channels, kernel_size=1),
            nn.Sigmoid() # Forces output to [0, 1] probability heatmap
        )

    def _conv_block(self, in_c, out_c):
        """
        Double convolution block with Batch Normalization.
        @param in_c: (int) input channels.
        @param out_c: (int) output channels.
        @return: (nn.Sequential) the block.
        """
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: (torch.Tensor) Input cube (Batch, 1, 32, 32, 32).
        @return: (torch.Tensor) Heatmap (Batch, 1, 32, 32, 32).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder with Skip Connections
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1) # The "Skip Connection"
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # The "Skip Connection"
        d1 = self.dec1(d1)
        
        # Pass the final decoder output (d1) to the final layer
        logits = self.final(d1)
        return logits
