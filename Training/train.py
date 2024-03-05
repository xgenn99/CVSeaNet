import os
import configparser
from prints.prints_fun import prints 
from Model.Model_light import CVSeaNet_light
from Dataset.DataModule import MS3DataModule
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import warnings, rasterio

if __name__ == "__main__":

# WARNINGS 
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    
# PARAMETERS DEFINITION FROM CONFIG
    config = configparser.ConfigParser()
    global_dir = os.path.abspath(os.getcwd())
    config.read(os.path.abspath(os.path.join(global_dir, 'Config/config.ini')))
    
    DATA_PATH = [config['General']['data_dir_sen'], config['General']['data_dir_csk']]
    ANNOTATIONS_PATH = [config['General']['annotations_dir_sen'], config['General']['annotations_dir_csk']]
    NUM_WORKERS = int(config['General']['num_workers'])
    STATE_PATH = config['General']['model_state_dir']
    SEED = int(config['General']['seed'])
    
    RESOLUTION = int(config['Input Data']['resolution'])
    IN_CHANNELS = int(config['Input Data']['in_channels'])
    TRANSFORM = config.getboolean('Input Data', 'transform')
    DTYPE = eval(config['Input Data']['dtype'])
    
    REAL_CONV_BLOCK = config.getboolean('Backbone', 'real_conv')
    MODEL = int(config['Backbone']['model_id'])
    
    if REAL_CONV_BLOCK:
        ARCHITECTURE_BACK = eval(config['Backbone']['architecture_real'])
    else:
        ARCHITECTURE_BACK = eval(config['Backbone']['architecture_comp'])

    BACKBONE_PARAMS = [MODEL, ARCHITECTURE_BACK]

    DATA_FUSION = config.getboolean('Data fusion', 'data_fusion_bool')
    DATA_FUSION_MODE = config['Data fusion']['data_fusion_mode']
    EARLY_DATA_FUSION = config.getboolean('Data fusion', 'early_data_fusion_bool')
    LATE_DATA_FUSION = config.getboolean('Data fusion', 'late_data_fusion_bool')
    
    ALPHA = float(config['Loss']['alpha'])
    GAMMA = float(config['Loss']['gamma'])
    WEIGHT_HM = float(config['Loss']['weight_hm'])
    WEIGHT_OFF = float(config['Loss']['weight_off'])

    SCORE_THRESHOLD = float(config['Accuracy']['score_threshold'])

    BATCH_SIZE = int(config['Training']['batch_size'])
    LR = float(config['Training']['learning_rate'])
    EPOCHS = int(config['Training']['epochs'])

    assert IN_CHANNELS == 1, "in_channels must be 1 because only one polarization is considered"
# PRE-TRAINING
    # torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("tb_logs", name="fist_training_v0")
    pl.seed_everything(SEED)
    if DTYPE == torch.float16:
        precision = '16-true'
    else:
        precision = '32-true'
        DTYPE = torch.float32

# DATA MODULE
    dm = MS3DataModule(data_path=DATA_PATH, coco_path=ANNOTATIONS_PATH, batch_size=BATCH_SIZE, real_conv_block=REAL_CONV_BLOCK,
                       num_workers=NUM_WORKERS, transform=TRANSFORM, dtype=DTYPE)

# MODEL    
    model = CVSeaNet_light(in_channels=IN_CHANNELS, in_resolution=RESOLUTION, backbone=BACKBONE_PARAMS,
                      real_conv_block=REAL_CONV_BLOCK,
                      score_threshold=SCORE_THRESHOLD,
                      data_fusion=DATA_FUSION, early_data_fusion=EARLY_DATA_FUSION,
                      late_data_fusion=LATE_DATA_FUSION, data_fusion_mode=DATA_FUSION_MODE,
                      alpha=ALPHA, gamma=GAMMA, weight_hm=WEIGHT_HM,  weight_off=WEIGHT_OFF, lr=LR)

# PRINTS
    prints(REAL_CONV_BLOCK, RESOLUTION, IN_CHANNELS, BATCH_SIZE, EPOCHS, LR, GAMMA, ALPHA, WEIGHT_HM, WEIGHT_OFF,
           SCORE_THRESHOLD)

# TRAINER
    trainer = pl.Trainer(logger=logger, max_epochs=EPOCHS, accelerator="gpu", precision=precision, log_every_n_steps=1)
    trainer.fit(model=model, datamodule=dm)
    # trainer.validate(model, dm)
    trainer.test(model, dm)

    
