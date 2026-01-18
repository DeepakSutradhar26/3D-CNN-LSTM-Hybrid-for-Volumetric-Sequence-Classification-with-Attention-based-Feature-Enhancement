from torch.optim import nadam
import torch.nn as nn
import tqdm
import os

import preprocess.normalization as pn
import metrics

def train():
    pn.normalize_train()
    pn.normalize_test()

    

def test():
    pass