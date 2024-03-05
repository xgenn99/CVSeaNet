from torch import nn
import torch
from torchvision.ops import focal_loss

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CustomLoss(nn.Module):

     def __init__(self, alpha: float = 0.5, gamma: float = 2.0, weight_hm: float = 0.5, weight_off: float = 0.5):
          super().__init__()

          self.weight_hm = weight_hm
          self.weight_off = weight_off
          self.alpha = alpha
          self.gamma = gamma

          self.hm_loss = CentNet_focal_loss(alpha=alpha, gamma=gamma)
          # self.hm_loss = nn.BCELoss(reduction='sum')
          # self.hm_loss = nn.SmoothL1Loss(reduction='sum')
          self.off_loss = nn.SmoothL1Loss(reduction='none')


     def forward(self, x_hm, x_off, x_hm_truth, x_off_truth):

          # print(f"\n preds idx: {torch.nonzero(x_hm > 0.5)} \ntruth idx: {torch.nonzero(x_hm_truth)}\n missed prob: {x_hm[torch.nonzero(x_hm_truth).unbind(1)]}")
          
          # hm_loss = self.hm_loss(x_hm, x_hm_truth)
          hm_loss = self.hm_loss(x_hm, x_hm_truth.to(dtype=x_hm.dtype))

     
          off_loss_masked = self.off_loss(x_off, x_off_truth)*x_hm_truth
          off_loss = torch.sum(off_loss_masked[off_loss_masked != 0])

          # print(self.weight_hm * hm_loss, self.weight_off * off_loss)


          return self.weight_hm * hm_loss + self.weight_off * off_loss
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CentNet_focal_loss(nn.Module):

     def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
          super().__init__()

          self.alpha = alpha
          self.gamma = gamma
          self.eps = 1e-5

          # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

     
     def forward(self, hm_preds, hm_truth):

          hm_preds, hm_truth  = hm_preds.to(torch.float64), hm_truth.to(torch.float64)
          # x_hm = self.max_pool(hm_preds)
          # # print(f"after pooling : {x_hm}")
          # mask = hm_preds == x_hm
          # # print(f"mask : {mask}")
          # hm_preds = mask * hm_preds
          # # print(f"masked out: {x_preds_hm}")

          pos_weights = hm_truth.eq(1)
          neg_weigths = hm_truth.eq(0)
          alpha = torch.where(hm_truth == 1, self.alpha, 1 - self.alpha)
          
          pos_loss = (hm_preds + self.eps).log()*(1 - hm_preds).pow(self.gamma)*pos_weights
          neg_loss = (1 - hm_preds + self.eps).log()*hm_preds.pow(self.gamma)*neg_weigths         

          # print((alpha*pos_loss).sum(), (alpha*neg_loss).sum())
          
          fl = alpha * (pos_loss + neg_loss)
          # fl = pos_loss + neg_loss
          # print(f"loss: {pos_loss.sum(), neg_loss.sum()}")

          # return - torch.mean(fl)
          # print(- torch.sum(fl))
          return - torch.sum(fl)