import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
from trainer import *
from models.Flat_MoE import *
from utils.dice_loss import *
from dataset import *
import pickle

with open("config.json", "r") as file:
    config = json.load(file)

LEARNING_RATE = config["LEARNING_RATE"]
DEVICE = config["DEVICE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
NUM_WORKERS = config["NUM_WORKERS"]
IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
IMAGE_WIDTH = config["IMAGE_WIDTH"]
PIN_MEMORY = config["PIN_MEMORY"]
LOAD_MODEL = config["LOAD_MODEL"]
TRAIN_IMG_DIR = config["TRAIN_IMG_DIR"]
TRAIN_MASK_DIR = config["TRAIN_MASK_DIR"]
VAL_IMG_DIR = config["VAL_IMG_DIR"]
VAL_MASK_DIR = config["VAL_MASK_DIR"]


DATASET_DIR = '../data/carvana-image-masking-challenge/'
WORKING_DIR = '../data/carvana-image-masking-challenge/'


train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

model = Flat_MoE(3, 3).to(DEVICE)
# loss_fn1 = lambda x, y: 0
loss_fn1 = nn.BCEWithLogitsLoss()
loss_fn2 = MutuallyExclusiveLoss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    BATCH_SIZE,
    train_transform,
    val_transform,
    NUM_WORKERS,
    PIN_MEMORY
)

if LOAD_MODEL:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
    check_accuracy(val_loader, model, device=DEVICE)

scaler = torch.amp.GradScaler('cuda')
losses = []
for epoch in range(NUM_EPOCHS):

    loss = train_fn(train_loader, model, optimizer, loss_fn1, loss_fn2, scaler)
    losses.append(loss)
    # save model
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)

    # print some examples to a folder
    save_predictions_as_imgs(
        val_loader, model, folder=WORKING_DIR, device=DEVICE
    )

# with open("losses", "wb") as f:
#     pickle.dump(losses, f)
