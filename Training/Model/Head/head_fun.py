import torch
from torch import nn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Head(nn.Module):
    r"""Definition of the detection head
    """
    def __init__(self, in_channels: int, score_threshold: float, real_conv_block=False):
        super().__init__()

        self.real_conv_block = real_conv_block
        self.thres = score_threshold
        self.in_channels = in_channels

        self.block_hm = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256,
                                   kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),                    
            nn.Sigmoid()
        )
        
        self.block_offset = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, 
                                      kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 3, 1, 1),                    
            nn.Sigmoid()

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_points = self.block_hm(x)
        x_off = self.block_offset(x)

        return x_points, x_off
