from torch.utils.data import DataLoader
from dataset import MRIDataset
import metrics
import os

train_loader = DataLoader(
    MRIDataset(os.path.join(metrics.NDATA_DIR, "train")),
    batch_size = metrics.BATCH_SIZE,
    num_workers = metrics.NUM_WORKERS,
    shuffle = True,
    pin_memory = True,
)

test_loader = DataLoader(
    MRIDataset(os.path.join(metrics.NDATA_DIR, "test")),
    batch_size = metrics.BATCH_SIZE,
    num_workers = metrics.NUM_WORKERS,
    shuffle = True,
    pin_memory = True,
)