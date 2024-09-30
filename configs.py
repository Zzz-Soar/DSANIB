# The configs of DSANIB 
from yacs.config import CfgNode as CN

cfg = CN()

cfg.DRUG = CN()
cfg.DRUG.NODE_IN_FEATS = 75
cfg.DRUG.PADDING = True
cfg.DRUG.GCN_Dimension = [128, 128, 128]
cfg.DRUG.NODE_EMBEDDING = 128
cfg.DRUG.MAX_NODES = 290

cfg.PROTEIN = CN()
cfg.PROTEIN.CNN_Dimension = [128, 128, 128]
cfg.PROTEIN.FILTER_SIZE = [3, 6, 9]
cfg.PROTEIN.EMB_SIZE = 128
cfg.PROTEIN.PADDING = True

cfg.MLPLayer = CN()
cfg.MLPLayer.IN_DIM = 256
cfg.MLPLayer.HIDDEN_DIM = 512
cfg.MLPLayer.OUT_DIMs = 256
cfg.MLPLayer.OUTPROB = 1

cfg.SETUP = CN()
cfg.SETUP.MAX_EPOCH = 100
cfg.SETUP.BATCH_SIZE = 32
cfg.SETUP.NUM_WORKERS = 0
cfg.SETUP.LR = 5e-5
cfg.SETUP.SEED = 42

# RESULT
cfg.RESULT = CN()
cfg.RESULT.PATH= "./result"

def get_cfg_defaults():
    return cfg.clone()
