import torch
from torch import nn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Head(nn.Module):
    r"""Definition of the detection head
    Arguments:
    - in_channels (int): the channels in output from the backbone
    - architecture (list of tuples and strings): a list that sums up the architecture in the form (out_channels, kernel_size, stride, padding)
    - BlockType (Any): the nn.Module class that is used for the head block
    - real_conv_block (bool)=False: if True, it works with real convolutions in the Backbone
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
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            
      
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
            # nn.BatchNorm2d(1),
            nn.Sigmoid()

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_points = self.block_hm(x)
        x_off = self.block_offset(x)

        # x_points = x_points * (x_points > self.thres)

        # x_hm = self.max_pool(x_points)
        # # print(f"after pooling : {x_hm}")
        # mask = x_points == x_hm
        # # print(f"mask : {mask}")
        # x_preds_hm = mask * x_points
        # print(f"masked out: {x_preds_hm}")

        # x_hm_suppressed = x_preds_hm * (x_preds_hm > self.thres)
        
        # # print(f"out suppressed with threshold: {x_hm_suppressed}")

        return x_points, x_off
        # return x_hm_suppressed, x_off
