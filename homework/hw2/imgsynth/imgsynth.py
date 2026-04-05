import os
from torch.utils.data import Dataset
from torchvision import transforms 
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# '/Users/maxwell/Spring26/Neural-Networks/homework/hw2/resources'

class CustomDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.total_imgs = len(os.listdir(main_dir))
        self.main_dir = main_dir
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def get_transform(mode:str):
    if mode ==  "simple":
        transform = transforms.Compose([
            transforms.Resize(opts.image_size, Image.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
    elif mode == "delux":
        transform = transforms.Compose([
            transforms.ColorJitter((1,2),(1,2),(1,2),(1,2))
        ])
        print("done")
    else: 
        raise NotImplementedError

    return transform

Class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        ######################################
        ## FILL IN CREATE ARCHITECTURE
        ######################################
        # self.conv1 = nn.Conv2d(...)
        # self.conv2 = nn.Conv2d(...)
        # self.conv3 = nn.Conv2d(...)
        # self.conv4 = nn.Conv2d(...)
        # self.conv5 = nn.Conv2d(...)

        # add norm layers when necessary

        # self.bn1 =
        # self.bn2 =
        # self.bn3 =
        # self.bn4 =

    def forward(self, z):
        """ Generate an image given a sample of random noise
        Input
        ----
            z: BS x noise_size x 1 x 1 --> 16x100x1x1
        Output
        ----
            out: BS x channels x image_width x image_height
        """
        out = F.relu(self.bn1(self.conv1(x)))

        ## FORWARD PASS

        out = self.conv5(out).squeeze()
        return out

if __name__ == "__main__": 
    print("asfddsa")
