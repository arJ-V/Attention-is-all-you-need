import torch

# --- General ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# --- Model ---
D_MODEL = 512
D_INNER = 2048
N_LAYERS = 6
N_HEAD = 8
D_K = 64
D_V = 64
DROPOUT = 0.1
N_POSITION = 200

# --- Data ---
SRC_VOCAB_SIZE = 10000 # Example size
TRG_VOCAB_SIZE = 10000 # Example size
SRC_PAD_IDX = 0
TRG_PAD_IDX = 0

# --- Sharing ---
TRG_EMB_PRJ_WEIGHT_SHARING = True
EMB_SRC_TRG_WEIGHT_SHARING = True
