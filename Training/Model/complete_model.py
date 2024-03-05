import torch, torchvision
from torch import nn
from Model.Backbone.backbone_fun import Backbone
from Model.Head.head_fun import Head

class CVSeaNet(nn.Module):
    """Defines the MS3 OD net
    Arguments:
    - in_channels (int): the input channels of the image
    - in_resolution (int): the resolution of the starting image
    - backbone_params (list): list of [model number, architecture]
    - head_params (list): list of 
    [architecture_head_hm, architecture_head_off ,blocktype]
    - real_conv_block(bool)=False: it activates the real convolution backbone,
    - data_fusion(bool)=False: if True activates data fusion
    - early_data_fusion(bool)=False: if True activates early data fusion
    - late_data_fusion(bool)=False: if True activates late data fusion
    - data_fusion_mode="cat": it can be "cat", "+", "*" dependending on which modality we want to fuse data with
    """
    def __init__(self, in_channels: int, in_resolution: int,
                backbone_params: list[int, list],
                score_threshold: float,
                real_conv_block: bool = False,
                data_fusion: bool = False, early_data_fusion: bool = False, late_data_fusion: bool = False, data_fusion_mode = "cat"
                ):
        super().__init__()

        self.in_channels = in_channels
        self.in_resolution = in_resolution
        self.real_conv_block = real_conv_block
        
        model_type = backbone_params[0]
        back_architecture = backbone_params[1]
        self.score_threshold = score_threshold

        if data_fusion:

            if early_data_fusion == late_data_fusion:

                raise ValueError(f"One value among early and late data fusion must be True and the other False: instead you got \nearly_data_fusion: {early_data_fusion} and late_data_fusion: {late_data_fusion[0]}")
        else:

            if early_data_fusion or late_data_fusion:

                raise ValueError(f"both early data fusion and late data fusion must be False. Instead you got \nearly_data_fusion: {early_data_fusion} and late_data_fusion: {late_data_fusion[0]}")  
            
        self.data_fusion = data_fusion
        self.early_df_bool = early_data_fusion
        self.late_df_bool = late_data_fusion

        if not self.real_conv_block:    
            in_channels_head = int(back_architecture[-1][1]*2)
        else:
            in_channels_head = int(back_architecture[-1][1])
        
        if self.data_fusion:

            self.df_mode = data_fusion_mode

            if self.early_df_bool:

                if self.real_conv_block:

                    if self.df_mode == "cat":

                        self.in_channels += 1
            
            if self.late_df_bool:

                if self.df_mode == "cat":

                    in_channels_head += 1        
   
 
        
        self.backbone = Backbone(in_channels=self.in_channels, architecture=back_architecture, model=model_type,
                                real_conv_block=self.real_conv_block)
        self.head = Head(in_channels=in_channels_head, score_threshold=self.score_threshold,
                real_conv_block=self.real_conv_block)
        # resnet = torchvision.models.resnet50()
        # resnet.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # convolutional_part = nn.Sequential(*list(resnet.children())[:-2])
        # self.backbone = convolutional_part
        # self.head = Head(in_channels=2048, score_threshold=self.score_threshold,
        #                 real_conv_block=self.real_conv_block)

        
        print(self.backbone)
        print(self.head)
        
    def forward(self, x: torch.Tensor, inc_angle: torch.Tensor = None):

        if self.data_fusion:

            inc_angle = inc_angle.unsqueeze(-1).unsqueeze(-1)
        
        if self.early_df_bool:
                
            inc_angle_early = inc_angle.expand(size=(x.shape[0], 1, x.shape[2], x.shape[3]))

        if self.real_conv_block:
            
            # x_real = x[:, :self.in_channels, ...]
            # x_imag = x[:, self.in_channels:, ...]

            # x = torch.sqrt(x_real **2 + x_imag **2)
        
            if self.early_df_bool:
        
                if self.df_mode == "cat":

                    x = torch.cat(tensors=(x, inc_angle_early), dim=1)
                
                elif self.df_mode == "+":

                    x = x + inc_angle_early

                elif self.df_mode == "*":

                    x = x * inc_angle_early

                else:

                    raise ValueError("The mode for early data fusion can be only (cat, +, *)") 
        
        else:
        
            if self.early_df_bool:

                if self.df_mode == "cat":

                    raise ValueError("cat mode not available for early complex data fusion")
                
                elif self.df_mode == "+":

                    x = x + inc_angle_early

                elif self.df_mode == "*":

                    x = x * inc_angle_early
                
                else:

                    raise ValueError("The mode for early data fusion can be only strings of (cat, +, *)") 
        
        # print(f"input: {x}")
        x = self.backbone(x)
        check_nan(x)

        # print(f"\n out back {x}") 
        # print(f"\nx shape after backbone {x.shape}")

        if self.late_df_bool:

            inc_angle_late = inc_angle.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        
            if self.df_mode == "cat":

                x = torch.cat(tensors=(x, inc_angle_late), dim=1)
            
            elif self.df_mode == "+":

                x = x + inc_angle_late

            elif self.df_mode == "*":

                x = x * inc_angle_late

            else:

                raise ValueError("The mode for early data fusion can be only (cat, +, *)") 

        x_hm_out, x_off_out = self.head(x)
        check_nan(x)

        # print(f"hm and hf shape after head: {x_hm_out.shape, x_off_out.shape}")
        # print(f"hm after head: {x_hm_out}")

        return {'Keypoints heatmap': x_hm_out, 'Offset heatmap': x_off_out, 'Scale Parameter': self.in_resolution/x_hm_out.shape[-1]}

def check_nan(x):
     if torch.isnan(x).any():
          raise ValueError("There is a nan")