from typing import Any
import torch
import pytorch_lightning as pl
from Model.complete_model import CVSeaNet
from Model.Loss.loss_fun import CustomLoss
from Model.Accuracy.OffsetAcc import MyAccuracy
import torchmetrics
from torch import nn
from check_nan import check_nan

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CVSeaNet_light(pl.LightningModule):
    def __init__(self, in_channels: int, in_resolution: int, backbone: list[int, list],
                score_threshold: float,
                real_conv_block: bool = False,
                data_fusion: bool = False, early_data_fusion: bool = False, late_data_fusion: bool = False, data_fusion_mode: str = "cat",
                alpha: float = 0.5, gamma : float = 2.0, weight_hm: float = 0.5, weight_off: float = 0.5, lr: float = 0.001):
        super().__init__()

        self.in_channels = in_channels
        self.in_resolution = in_resolution
        self.backbone = backbone
        self.score_threshold = score_threshold
        self.real_conv_block = real_conv_block
        self.data_fusion = data_fusion
        self.early_df = early_data_fusion
        self.late_df = late_data_fusion
        self.df_mode = data_fusion_mode

        self.alpha = alpha
        self.gamma = gamma
        self.weight_hm = weight_hm
        self.weight_off = weight_off

        self.lr = lr

        hparams = {
            "in_channels": self.in_channels,
            "in_res": self.in_resolution,
            "backbone_type_id": self.backbone[0],
            "backbone_architecture": self.backbone[1],
            "score_threshold": self.score_threshold,
            "real_conv_block": self.real_conv_block,
            "data_fusion": self.data_fusion,
            "early_data_fusion": self.early_df,
            "late_data_fusion": self.late_df,
            "data_fusion_mode": self.df_mode,
            "weight_alpha_Focloss": self.alpha,
            "weight_gamma_Focloss": self.gamma,
            "weigh_hm_loss": self.weight_hm,
            "weight_off_loss": self.weight_off,
            "lr": self.lr

        }


        self.save_hyperparameters(hparams)

        self.model = CVSeaNet(in_channels=self.in_channels, in_resolution=self.in_resolution,
                        backbone_params=self.backbone,
                        score_threshold=self.score_threshold, real_conv_block=self.real_conv_block,
                        data_fusion=self.data_fusion, 
                        early_data_fusion=self.early_df, late_data_fusion=self.late_df,
                        data_fusion_mode=self.df_mode)
        
        self.loss_fun = CustomLoss(alpha=self.alpha, gamma=self.gamma,
                                   weight_hm=self.weight_hm, weight_off=self.weight_off)
        

        self.accuracy = torchmetrics.Accuracy(task='binary', threshold=self.score_threshold)
        self.hm_f1 = torchmetrics.F1Score(task='binary', threshold=self.score_threshold)
        self.hm_prec = torchmetrics.Precision(task='binary', threshold=self.score_threshold)
        self.hm_recall = torchmetrics.Recall(task='binary', threshold=self.score_threshold)
        self.off_metrics = MyAccuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):

        loss, y_preds, y_truth, off_preds, off_truth = self.common_box(batch)
        
        acc_hm = self.accuracy(y_preds, y_truth)
        acc_offset = self.off_metrics(off_preds, off_truth, y_truth)
        f1 = self.hm_f1(y_preds, y_truth)
        prec = self.hm_prec(y_preds, y_truth)
        rec = self.hm_recall(y_preds, y_truth)

        metrics = {'train_loss': loss, "train_acc": acc_hm, "train_precision": prec,
                   "train_mse_offs": acc_offset, "train_rec": rec,
                   "train_f1": f1}
        
        self.log_dict(metrics,
                       on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        
        loss, y_preds, y_truth, off_preds, off_truth = self.common_box(batch)
        
        acc_hm = self.accuracy(y_preds, y_truth)
        f1 = self.hm_f1(y_preds, y_truth)
        prec = self.hm_prec(y_preds, y_truth)
        rec = self.hm_recall(y_preds, y_truth)
        off_acc = self.off_metrics(off_preds, off_truth, y_truth)

        metrics = {'test_loss': loss, "test_acc": acc_hm,
                   "test_prec": prec, "test_rec": rec,
                   "test_f1": f1, "test_mse_offs": off_acc}
        
        self.log_dict(metrics,
                       on_step=True, on_epoch=True, logger=True, prog_bar=True)
        
        return metrics

    def forward(self, x, z):
        
        return self.model(x, z)
    
    def common_box(self, batch):
        
        x, y_truth, inc_angle, off_truth = batch

        output = self.model(x, inc_angle)
        
        y_preds = output['Keypoints heatmap']
        off_preds = output['Offset heatmap']
        loss = self.loss_fun(y_preds, off_preds, y_truth, off_truth)

        check_nan(x)
        check_nan(y_truth)
        check_nan(inc_angle)
        check_nan(off_truth)

        check_nan(y_preds)
        check_nan(off_preds)
        check_nan(loss)

        return loss,  y_preds, y_truth, off_preds, off_truth
