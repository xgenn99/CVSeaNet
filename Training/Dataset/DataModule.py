import torch
from Dataset.CustomDataset_sen import MS3Dataset_SEN
from Dataset.CustomDataset_csk import MS3Dataset_CSK
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from tqdm import tqdm

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def custom_collate(batch):
    """The collate function defines how batch is assembled, as the output of DataLoader
    Returns:
    - img stacked (shape of bs x channs x input_size x input_size)
    - target heatmap with all zeros apart from the ones in the indices = target centroids 
    (shape of bs x 1 x out_size x out_size)
    - inc angle for each img in the batch (shape of bs x 1)
    - offset grid in which there are all 0 apart from the indices = target centroids scaled in which there is
    the value of the offset (shape of bs x 2 x out_size x out_size)
    """

    shape_img = 4  #CHANGE HERE this is the shape of the output layer 16-512, 32-1024, 64-2048
    scale = batch[0]["sample"].shape[-1]/shape_img
    dtype = batch[0]["sample"].dtype

    IMG = []
    INC_ANG = []
    TGT_HM = torch.zeros(size=(len(batch), shape_img, shape_img))
    TGT_OFFSET = torch.zeros(size=(len(batch), 2, shape_img, shape_img))

    for i in range(len(batch)):
        
        img, target, inc_angle = batch[i]["sample"], batch[i]["target"], batch[i]["inc_angle"]
        IMG.append(img)
        target_scaled = torch.floor(target/scale).to(dtype=torch.int)

        for s in range(target_scaled.shape[0]):

            x_scaled = target_scaled[s, 0]
            y_scaled = target_scaled[s, 1]
            if target_scaled[s, 0] == shape_img:
                x_scaled = target_scaled[s, 0] - 1
            if target_scaled[s,1] == shape_img:
                y_scaled = target_scaled[s, 1] - 1

            TGT_HM[i, x_scaled.item(), y_scaled.item()] = 1
            TGT_OFFSET[i, 0, x_scaled.item(), y_scaled.item()] = torch.abs(target[s,0]/scale - x_scaled)
            TGT_OFFSET[i, 1, x_scaled.item(), y_scaled.item()] = torch.abs(target[s,1]/scale - y_scaled)

        INC_ANG.append(inc_angle)

    return torch.stack(IMG).to(dtype=dtype), TGT_HM.unsqueeze(1).to(dtype=torch.bool), torch.stack(INC_ANG).to(dtype=dtype), TGT_OFFSET.to(dtype=dtype)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MS3DataModule(pl.LightningDataModule):
    def __init__(self, data_path, coco_path, batch_size: int, real_conv_block: bool,
                num_workers: int, transform: bool =  True,
                dtype = torch.float32) :
        super().__init__()

        self.data_path = data_path
        self.coco_path = coco_path
        self.bs = batch_size
        self.num_workers = num_workers
        self.trans = transform
        self.dtype = dtype
        self.real_conv_block = real_conv_block

    def setup(self, stage=None):
        ds_sen = MS3Dataset_SEN(data_path=self.data_path[0], coco_path=self.coco_path[0], real_conv=self.real_conv_block,
                        transform=self.trans, dtype=self.dtype)
        ds_csk = MS3Dataset_CSK(data_path=self.data_path[1], coco_path=self.coco_path[1], real_conv=self.real_conv_block,
                        transform=self.trans, dtype=self.dtype)
        ds = ConcatDataset([ds_sen, ds_csk])
        
        # self.train_ds ,self.val_ds, self.test_ds = random_split(ds, [0.6, 0.2, 0.2])
        self.train_ds, self.test_ds = random_split(ds, [0.8, 0.2])
        # self.train_ds, self.test_ds = random_split(ds, [0.0002, 0.9998])

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        return DataLoader(dataset=self.train_ds, batch_size=self.bs, num_workers=self.num_workers,
                           shuffle=True, collate_fn=custom_collate)
    
    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(dataset=self.val_ds, batch_size=self.bs, num_workers=self.num_workers,
    #                        shuffle=False, collate_fn=custom_collate)
    
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.test_ds, batch_size=self.bs, num_workers=self.num_workers,
                           shuffle=False, collate_fn=custom_collate)
