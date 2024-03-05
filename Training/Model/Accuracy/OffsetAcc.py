import torch
from torchmetrics import Metric
from torch import nn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
class MyAccuracy(Metric):
  def __init__(self):
    super().__init__()
    self.l1_loss = nn.SmoothL1Loss(reduction='none')
    self.add_state("off_loss", default=torch.tensor(0))
    
  def update(self, x_off, x_off_truth, x_hm_truth) -> None:
    if x_off.shape != x_off_truth.shape:
      raise ValueError("preds and target must have the same shape")
    off_loss_masked = self.l1_loss(x_off, x_off_truth)*x_hm_truth
    
    self.off_loss = torch.mean(off_loss_masked[off_loss_masked != 0])

  def compute(self):
    return 1 - self.off_loss


