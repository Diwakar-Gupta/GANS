import os
import torch
import config
from torchvision.utils import save_image


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint at", filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint from", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_some_examples(model, dataset, folder, epoch):
    x, y = next(iter(dataset))
    x, y = x.to(config.device), y.to(config.device)
    model.eval()
    with torch.no_grad():
        y_fake = model(x)
        for i in range(min(4, x.shape[0])):
            img = torch.cat([x[i], y_fake[i], y[i]], 2)
            img = (img+1)*127   # remove normalization
            save_image(img, os.path.join(folder, f"gen_{epoch}_{i}.png"))
    model.train()