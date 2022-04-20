from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import MapDataset
from torch import optim
import config
import utils

def train_fn(disc, gen, train_loader, opt_disc, opt_gen, LI_LOSS, BCE):
    loop = tqdm(train_loader)
    
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)
        y_fake = gen(x)
        d_real = disc(x, y)
        d_fake = disc(x, y_fake.detach())
        d_real_loss = BCE(d_real, torch.ones_like(d_real))
        d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss+d_fake_loss)/2

        disc.zero_grad()
        d_loss.backward(retain_graph=True)
        opt_disc.step()

        d_fake = disc(x, y_fake)
        g_loss = BCE(d_fake, torch.ones_like(d_fake)) + LI_LOSS(y_fake, y)*config.L1_LAMBDA
        gen.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_gen.step()


def main():
    disc = Discriminator(in_channels=3).to(config.device)
    gen = Generator(in_channels=3).to(config.device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    LI_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        utils.load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    train_ds = MapDataset(config.DATASET_ROOT, train=True)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_ds = MapDataset(config.DATASET_ROOT, train=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, LI_LOSS, BCE)

        if config.SAVE_MODEL and epoch%5==0:
            utils.save_checkpoint(gen, opt_gen, config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, config.CHECKPOINT_DISC)
        
        utils.save_some_examples(gen, val_loader, 'evaluation/', epoch)
        


if __name__ == '__main__':
    main()