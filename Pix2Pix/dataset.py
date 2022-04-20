from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import random


both_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(1, 1, 1)),
    ]
)

class MapDataset(Dataset):
    def __init__(self, root_dir, train=True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.list_files = glob.glob(os.path.join(self.root_dir, 'train' if train else 'val', '*.jpg'))
        # if shuffle: # shuffling can be handled by DataLoader
        #     random.shuffle(self.list_files)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_path = self.list_files[index]
        # img_path = os.path.join(self.root_dir, img_path)
        image = np.array(Image.open(img_path))
        half = image.shape[1]//2

        input_image = Image.fromarray(image[:, :half, :]) # image width = 1200
        target_image = Image.fromarray(image[:, half:, :])
        
        # if self.root_dir.endswith('TI') or self.root_dir.endswith('TI/'):
        #     t = input_image
        #     input_image=target_image
        #     target_image=t


        input_image = both_transform(input_image)
        target_image = both_transform(target_image)

        input_image = (input_image-127.5)/127.5
        target_image = (target_image-127.5)/127.5
        
        return input_image, target_image


def test():
    dataset = MapDataset('../dataset/pixTopixTI', shuffle=True)
    (input, target) = next(iter(dataset))
    from torchvision.utils import save_image
    save_image((input+1)*127, 'input.jpg')
    save_image((target+1)*127, 'target.jpg')

if __name__ == '__main__':
    test()