import torch
from Model.complete_model import CVSeaNet
from Dataset.CustomDataset_csk import MS3Dataset_CSK
from Dataset.CustomDataset_sen import MS3Dataset_SEN
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

path_state = r"/media/warmachine/Seagate Expansion Drive/DATASET_MMMF/folder_to_work_on/tb_logs/fist_training_v0/version_11/checkpoints/epoch=64-step=134160.ckpt"
state = torch.load(path_state)
in_channels = in_channels = state['hyper_parameters']['in_channels']
in_res = state['hyper_parameters']['in_res']
backbone_type_id = state['hyper_parameters']['backbone_type_id']
backbone_architecture = state['hyper_parameters']['backbone_architecture']
score_threshold = state['hyper_parameters']['score_threshold']
real_conv_block = state['hyper_parameters']['real_conv_block']
data_fusion = state['hyper_parameters']['data_fusion']
early_data_fusion = state['hyper_parameters']['early_data_fusion']
late_data_fusion = state['hyper_parameters']['late_data_fusion']
data_fusion_mode = state['hyper_parameters']['data_fusion_mode']
model = CVSeaNet(in_channels=in_channels, in_resolution=in_res, backbone_params=[backbone_type_id, backbone_architecture],
                 score_threshold=score_threshold, real_conv_block=real_conv_block, data_fusion=data_fusion,
                 early_data_fusion=early_data_fusion, late_data_fusion=late_data_fusion, data_fusion_mode=data_fusion_mode).to(device='cuda', dtype=torch.float16)
state_dict_without_prefix = {key.replace('model.', ''): value for key, value in state['state_dict'].items()}     
model.load_state_dict(state_dict_without_prefix)
#fix model without incidence angle

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555
#missing part to load image -need to define a function with rasterio and trasform
img = img.unsqueeze(0).to(device='cuda', dtype=torch.float16)

model.eval()
out = model(img)
idx_kp = torch.nonzero(out['Keypoints heatmap'] > score_threshold)
idx_kp = idx_kp[:, 2:]
off = out['Offset heatmap']
x_off = off[0,0,idx_kp[:,0], idx_kp[:,1]].unsqueeze(-1)
y_off = off[0,1,idx_kp[:,0], idx_kp[:,1]].unsqueeze(-1)
off = torch.cat((x_off, y_off),dim=-1)
scaled_out = idx_kp + off
scaled_out = scaled_out * out['Scale Parameter']
plt.figure(figsize=(20,20))
plt.imshow((torch.abs(img[0,0].to(torch.float) + 1j * img[0,1].to(torch.float)).cpu().numpy()), cmap='gray')
plt.scatter(scaled_out[:,0].to(torch.float).detach().cpu().numpy(), scaled_out[:,1].to(torch.float).detach().cpu().numpy(), marker='x', c='blue')