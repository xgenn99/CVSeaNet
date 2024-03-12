from torch import nn
import torch

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CustomLoss(nn.Module):

     def __init__(self, alpha: float = 0.5, gamma: float = 2.0, weight_hm: float = 0.5, weight_off: float = 0.5):
          super().__init__()
          """Defines the loss function for training
          Arguments:
          - alpha (float) = 0.5: parameter in alpha-balanced focal loss
          - gamma(float) = 2.0: parameter in focal loss
          - weight_hm (float) = 0.5: weight for keypoints heatmap in loss function
          - weight_off(float) = 0.5 : weight for offset heatmap in the loss function
          """

          self.weight_hm = weight_hm
          self.weight_off = weight_off
          self.alpha = alpha
          self.gamma = gamma

          self.hm_loss = CentNet_focal_loss(alpha=alpha, gamma=gamma)
          self.off_loss = nn.SmoothL1Loss(reduction='none')


     def forward(self, x_hm, x_off, x_hm_truth, x_off_truth):
          hm_loss = self.hm_loss(x_hm, x_hm_truth.to(dtype=x_hm.dtype))

          off_loss_masked = self.off_loss(x_off, x_off_truth)*x_hm_truth
          off_loss = torch.sum(off_loss_masked[off_loss_masked != 0])
          
          return self.weight_hm * hm_loss + self.weight_off * off_loss
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CentNet_focal_loss(nn.Module):

     def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
          super().__init__()

          self.alpha = alpha
          self.gamma = gamma
          self.eps = 1e-5

     def forward(self, hm_preds, hm_truth):

          hm_preds, hm_truth  = hm_preds.to(torch.float64), hm_truth.to(torch.float64)
          pos_weights = hm_truth.eq(1)
          neg_weigths = hm_truth.eq(0)
          alpha = torch.where(hm_truth == 1, self.alpha, 1 - self.alpha)
          
          pos_loss = (hm_preds + self.eps).log()*(1 - hm_preds).pow(self.gamma)*pos_weights
          neg_loss = (1 - hm_preds + self.eps).log()*hm_preds.pow(self.gamma)*neg_weigths         
          
          fl = alpha * (pos_loss + neg_loss)
          
          return - torch.sum(fl)
