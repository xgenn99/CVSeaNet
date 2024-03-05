import torch, torchvision
from torch import nn
import einops as e
import math
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Backbone(nn.Module):
    r"""Backbone based on efficientNet, but applied on complex numbers
    """
    def __init__(self, in_channels: int, architecture: list, model: int, real_conv_block=False):
        super().__init__()
        
        self.in_channels = in_channels
        depth_factor, width_factor = self.phi_from_model(model=model)
        self.df = depth_factor
        self.wf = width_factor
        self.real_conv_block = real_conv_block

        self.ConvNet = self._create_layers(architecture=architecture)

    def forward(self, x: torch.Tensor):

        x = self.ConvNet(x)
        
        return x
    
    def phi_from_model(self, model: int):
        """Insert the model and find depth and width factors
        Arguments:
        - model (int): specifies which EfficientNet model it uses (from 0 to 7)
        """

        if model < 0 or model > 7:
            
            raise ValueError("The model input (int) must be in the range [0,7]")

        model_param = [[0, 1.2, 1.1],
                       [0.5, 1.2, 1.1],
                       [1, 1.2, 1.1],
                       [2, 1.2, 1.1],
                       [3, 1.2, 1.1],
                       [4, 1.2, 1.1],
                       [5, 1.2, 1.1],
                       [6, 1.2, 1.1]]
        
        phi, alpha, beta = model_param[model]
        depth_factor = int(alpha ** phi)
        width_factor = int(beta ** phi)
       
        return depth_factor, width_factor
    
    def _create_layers(self, architecture):
        r"""Creates the layers for the network"""
        
        layers = []
        back_in_channels = architecture[0]

        if not self.real_conv_block:

            initial_block = nn.Sequential(
                
                ComplexConv2d(in_channels=self.in_channels, out_channels=back_in_channels, kernel_size=3, stride=2,
                              padding=0, bias=False),
                ComplexReLU6(inplace=True),
            )
        
        else:
            
            initial_block = nn.Sequential(

                nn.Conv2d(in_channels=self.in_channels, out_channels=back_in_channels, kernel_size=3, stride=2,
                          padding=0, bias=False),
                nn.BatchNorm2d(num_features=back_in_channels),
                nn.ReLU6(inplace=True)
            )

        
        for x in architecture:
                
                if type(x) == tuple:
                
                    for _ in range(int(x[-1]*self.df)):
                        
                        layers += [ComplexMBConvBlock(in_channels=int(back_in_channels*self.wf), exp_factor=x[0],
                                                    out_channels=int(x[1]*self.wf),
                                                    kernel_size_dw=x[2], stride_dw=x[3], reduced_dim_se=4,
                                                    real_conv_block=self.real_conv_block, bias=False)]
                        
                        back_in_channels = int(x[1]*self.wf)
        
        return nn.Sequential(initial_block, *layers)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexMBConvBlock(nn.Module):
    """Defines the complex valued convolutional block, based on  MBConvBlock, e.g. Inverted Residual Block
    Arguments:
    - in_channels (int): the input channels
    - exp_factor (int) > 1: the value that multiplied by the input channels, gives the hidden units in the bottleneck configuration
    - out_channels (int): the output channels
    - kernel_size_dw (int): the kernel size for the depthwise convolution
    - stride_dw (int): the stride for the depthwise convolution
    - reduced_dim_se (int): the parameter that reduces the dimension of the channels in the squeeze excitation block
    - real_conv (bool)=False: it is True if the convolution is real
    - bias(bool)=True: it is True if biases are added in the block layers
    """
    ## hidden units must be bigger than input channels
    def __init__(self, in_channels: int, exp_factor: int, out_channels: int,
                kernel_size_dw: int, stride_dw: int, reduced_dim_se: int,
                real_conv_block=False, bias=True):
        super().__init__()
        
        self.in_feat = in_channels
        self.out_feat = out_channels
        self.kernel_size = kernel_size_dw
        self.stride = stride_dw
        self.padding = (self.kernel_size-1)//2
        self.real_conv_block = real_conv_block
        self.bias = bias


        assert exp_factor >= 1, "the expansion factor is smaller than 1. It must be much bigger"
         
        self.hidden_units = exp_factor * self.in_feat
        self.reduced_dim_se = int(self.hidden_units/reduced_dim_se)
        
        if self.real_conv_block == False:

            self.block = nn.Sequential(
                ComplexConv2d(in_channels=self.in_feat, out_channels=self.hidden_units,
                                            kernel_size=1, stride=1, padding=0, bias=self.bias),
                ComplexReLU6(inplace=True),
                nn.BatchNorm2d(self.hidden_units*2),
                ComplexDwConv2d(in_channels=self.hidden_units,kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, bias=self.bias),
                nn.BatchNorm2d(self.hidden_units*2),
                ComplexReLU6(inplace=True),
                ComplexSqueezeExcitation(in_channels=self.hidden_units, reduced_dim=self.reduced_dim_se),
                ComplexConv2d(in_channels=self.hidden_units, out_channels=self.out_feat,
                                            kernel_size=1, stride=1, padding=0, bias=self.bias),
                nn.BatchNorm2d(self.out_feat*2),
            )
            
        else:
            
            self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_feat, out_channels=self.hidden_units, kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.BatchNorm2d(self.hidden_units),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=self.hidden_units, out_channels=self.hidden_units,
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                        groups=self.hidden_units, bias=self.bias),
            nn.BatchNorm2d(self.hidden_units),
            nn.ReLU6(inplace=True),
            torchvision.ops.SqueezeExcitation(input_channels=self.hidden_units, squeeze_channels=reduced_dim_se),
            nn.Conv2d(in_channels=self.hidden_units, out_channels=self.out_feat,
                       kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.BatchNorm2d(self.out_feat),
        )
        
        if self.out_feat == self.in_feat and self.stride == 1:

            self.use_res = True
        
        else:
            
            self.use_res = False

    def forward(self, x: torch.Tensor):

        x_res = x

        x = self.block(x)

        if self.use_res:

            return torch.add(x, x_res)
        
        else:
            
            return x
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexConv2d(nn.Module):
    """Defines the complex convolutional layer
    Arguments:
    - in_channels (int): input channels
    - out_channels (int): output channels
    - kernel_size (int): kernel size
    - padding (int): padding
    - bias (bool)=True: if it is True bias are added in the layers 
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.convLayer_real = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
        self.convLayer_imag = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x_real = x[:, :self.in_channels,...]
        x_imag = x[:, self.in_channels:,...]

        real_part = self.convLayer_real(x_real) - self.convLayer_imag(x_imag)
        imag_part = self.convLayer_real(x_imag) + self.convLayer_imag(x_real)

        return torch.cat((real_part, imag_part), dim=1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexDwConv2d(nn.Module):
    """Defines the depthwise complex convolution layer
    Arguments:
    - in_channels (int): input channels
    - kernel_size (int): kernel size 
    - stride (int): stride 
    - padding (int): padding
    - bias (bool)=True: it is true if bias are added in the layers
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: int, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.s = stride
        self.f = kernel_size
        self.p = padding
        self.bias = bias

        self.real_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                    kernel_size=self.f, stride=self.s, padding=self.p, groups=in_channels, bias=self.bias)
        
        self.imag_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=self.f, stride=self.s, padding=self.p, groups=in_channels, bias=self.bias)
    
    def forward(self, x: torch.Tensor):

        x_real = x[:, :self.in_channels,...]
        x_imag = x[:, self.in_channels:,...] 

        real_part = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag_part = self.real_conv(x_imag) + self.imag_conv(x_real)
    
        return  torch.cat(tensors=(real_part, imag_part), dim=1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexSqueezeExcitation(nn.Module):
    """Defines squeeze excitation for complex numbers
    Arguments:
    - in_channels (int): input channels
    - reduced_dim (int): the parameter that reduces the dimension
    """
    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.red_dim = reduced_dim
    
        self.se = nn.Sequential(
            ComplexAdaptiveAvgPool2d(output_size=1),
            ComplexConv2d(in_channels=self.in_channels, out_channels=self.red_dim , kernel_size=1),
            ComplexSiLU(),
            ComplexConv2d(in_channels=self.red_dim , out_channels=self.in_channels, kernel_size=1),
            ComplexSigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x*self.se(x)
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexAdaptiveAvgPool2d(nn.Module):
    """Defines the Complex Adaptive AVg pool 2d needed for the Squeeze Excitation block
    Arguments:
    - output_size (int or tuple[int, int]): the output size for the adaptive avgpool
    """
    def __init__(self, output_size: int or tuple[int, int]):
        super().__init__()
        
        self.avgpool_real = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.avgpool_imag = nn.AdaptiveAvgPool2d(output_size=output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ind = int(x.shape[1]/2)
        x_real = x[:, :ind,...]
        x_imag = x[:,ind:,...] 

        real_part = self.avgpool_real(x_real)
        imag_part = self.avgpool_imag(x_imag)
        
        return torch.cat((real_part, imag_part), dim=1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexSiLU(nn.Module):
    """Define the Complex Silu
    Arguments:
    - in_place (bool)=False: the output of Silu occupies the same memory
    """
    def __init__(self, inplace=False):
        super().__init__()

        self.silu_real = nn.SiLU(inplace=inplace)
        self.silu_imag = nn.SiLU(inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        ind = int(x.shape[1]/2)
        x_real = x[:, :ind,...]
        x_imag = x[:,ind:,...] 
        
        real_part = self.silu_real(x_real)
        imag_part = self.silu_imag(x_imag)

        return torch.cat((real_part, imag_part), dim=1)

class ComplexReLU(nn.Module):
    """Defines the Complex ReLU"""
    
    def __init__(self, inplace=False):
        super().__init__()
        
        self.relu_real = nn.ReLU(inplace=inplace)
        self.relu_imag = nn.ReLU(inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        ind = int(math.ceil(x.shape[1]/2))
        x_real = x[:, :ind,...]
        x_imag = x[:,ind:,...] 
        
        real_part = self.relu_real(x_real)
        imag_part = self.relu_imag(x_imag) 

        return torch.cat((real_part, imag_part), dim=1)

class ComplexReLU6(nn.Module):
    """Defines the Complex ReLU"""
    
    def __init__(self, inplace=False):
        super().__init__()
        
        self.relu_real = nn.ReLU6(inplace=inplace)
        self.relu_imag = nn.ReLU6(inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ind = int(math.ceil(x.shape[1]/2))
        x_real = x[:, :ind,...]
        x_imag = x[:,ind:,...] 

        real_part = self.relu_real(x_real)
        imag_part = self.relu_imag(x_imag) 

        return torch.cat((real_part, imag_part), dim=1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexBatchNorm2d(nn.Module):
    """Defines the Complex Batch Normalization 2d"""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9) :
        super().__init__()

        self.num = num_features
        self.eps = eps
        self.momentum = momentum

        gamma_rr = 1/math.sqrt(2)
        gamma_ii = gamma_rr
        gamma_ri = 0
        gamma_ir = gamma_ri 
        self.gamma = nn.Parameter(torch.tensor([[gamma_rr, gamma_ri], [gamma_ir, gamma_ii]]).expand(num_features, -1, -1))
        self.beta = nn.Parameter(torch.zeros(num_features, 2, 1))        

        self.register_buffer('running_mean_r', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean_i', torch.zeros(1, num_features, 1, 1))
        
        cov_in = 1/math.sqrt(2)
        self.register_buffer('covRR', torch.ones(num_features)*cov_in)
        self.register_buffer('covII', torch.ones(num_features)*cov_in)
        self.register_buffer('covRI', torch.zeros(num_features))
        self.register_buffer('covIR', torch.zeros(num_features))


    def forward(self, x: torch.Tensor):

        x_real = x[:, :self.num,...]
        x_imag = x[:, self.num:,...] 
        dtype= x_real.dtype

        if self.training:

            covRR = self.cov(x_real, x_real)[0]
            covII = self.cov(x_imag, x_imag)[0]
            covRI, x_cent_real, x_cent_imag, mean_r, mean_i = self.cov(x_real, x_imag)
            covIR = self.cov(x_imag, x_real)[0]

            self.running_mean_r = (1 - self.momentum) * self.running_mean_r + self.momentum * mean_r
            self.running_mean_i = (1 - self.momentum) * self.running_mean_i + self.momentum * mean_i

            self.covRR, self.covII = (1 - self.momentum) * self.covRR + self.momentum * covRR, (1 - self.momentum) * self.covII + self.momentum * covII
            
            self.covRI, self.covIR = (1 - self.momentum) * self.covRI + self.momentum * covRI, (1 - self.momentum) * self.covIR + self.momentum * covIR

            eps_mat = torch.eye(2).expand(self.num, -1, -1).to(device=x_real.device) * self.eps
            cov_matrix = torch.stack((torch.stack((covRR, covRI), dim=-1), torch.stack((covIR, covII), dim=-1)), dim=-1)
            inverse_V = torch.inverse(self.sqrtM(cov_matrix + eps_mat).to(dtype=torch.float)).to(dtype=dtype)

            X_exp = e.rearrange(torch.stack((x_cent_real, x_cent_imag), dim=-1).unsqueeze(-1), 'b c h w t o -> b h w c t o')
            x_tilde = torch.matmul(inverse_V, X_exp)

            x_bn = torch.matmul(self.gamma, x_tilde) + self.beta
            x_bn = e.rearrange(x_bn,' b h w c t o -> b c h w t o').squeeze(-1)

            x_bn_real, x_bn_imag = x_bn[...,0], x_bn[...,1]

        else:

            x_cent_real = x_real - self.running_mean_r
            x_cent_imag = x_imag - self.running_mean_i
            
            cov_matrix = torch.stack((torch.stack((self.covRR, self.covRI), dim=-1), torch.stack((self.covIR, self.covII), dim=-1)), dim=-1)
            inverse_V = torch.inverse(self.sqrtM(cov_matrix + eps_mat))

            X_exp = e.rearrange(torch.stack((x_cent_real, x_cent_imag), dim=-1).unsqueeze(-1), 'b c h w t o -> b h w c t o')
            x_tilde = torch.matmul(inverse_V, X_exp)

            x_bn = e.rearrange(self.gamma * x_tilde + self.beta,' b h w c t o -> b c h w t o').squeeze(-1)
            x_bn_real, x_bn_imag = x_bn[...,0], x_bn[...,1]

        return torch.cat((x_bn_real, x_bn_imag), dim=1)


    @staticmethod
    def cov(x: torch.Tensor, y: torch.Tensor):

        assert x.shape == y.shape

        x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
        x_cent = x - x_mean
        y_cent = y - y_mean

        n = ((x - x_mean) * ((y - y_mean))).sum(dim=(0,2,3))
        d = x.numel()/x.shape[1] - 1

        return n/d, x_cent, y_cent, x_mean, y_mean
    

    def sqrtm(self, mat: torch.Tensor):

        tau = mat[0, 0] + mat[1, 1]
        delta = torch.det(mat.to(dtype=torch.float))

        s = torch.sqrt(delta)
        t = torch.sqrt(tau +2*delta)

        S = s.expand_as(mat)

        return (mat + S)/t
    
    def sqrtM(self, mat):

        M = torch.zeros(self.num, 2, 2)

        for i in range(self.num):
            
            m = mat[i]

            M[i] = self.sqrtm(m)

        return M.to(device=mat.device)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ComplexSigmoid(nn.Module):
    """Defines the Complex Sigmoid

    """
    def __init__(self):
        super().__init__()
        
        self.sigm = torch.nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        ind = int(math.ceil(x.shape[1]/2))
        x_real = x[:, :ind,...]
        x_imag = x[:,ind:,...] 

        real_part = self.sigm(x_real)
        imag_part = self.sigm(x_imag)

        return torch.cat((real_part, imag_part), dim=1)
