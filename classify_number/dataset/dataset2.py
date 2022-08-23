from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
from albumentations import Compose, HueSaturationValue, RandomBrightnessContrast, OneOf, IAAAdditiveGaussianNoise, \
    MotionBlur, GaussianBlur, ImageCompression, GaussNoise, Resize, RandomCrop, RandomRotate90, RandomGridShuffle

# from albumentations import *
# celeb[0.39, 0.28, 0.27], [0.21, 0.16, 0.15]
# ff [0.44,0.38,0.39], [0.23,0.2,0.21]
data_transform = {
            "train": transforms.Compose([  # transforms.RandomCrop(240),
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.RandomErasing(0.7,scale=(0.05,0.2),ratio=(0.5,2)),
                # transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
            ]),
            "val": transforms.Compose([
                # transforms.CenterCrop(240),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                # transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
            ])
        }


class TestDataset(Dataset):
    def __init__(self, label_file, root_dir, phase=None):

        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.pahse = phase

    #  return the number of samples in dataset
    def __len__(self):
        return len(self.labels)

    #  return a sample at the given index
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.pahse == 'train':
            tfms = data_transform['train']
        else:
            tfms = data_transform['val']
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = tfms(img).unsqueeze(0)
        label = self.labels.iloc[index, 1]
        img = img.squeeze(0)
        

        return img, label, img_path


class BinaryDataset(Dataset):
    def __init__(self, label_file, root_dir, phase=None):

        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.pahse = phase

    #  return the number of samples in dataset
    def __len__(self):
        return len(self.labels)

    #  return a sample at the given index
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.pahse == 'train':
            aug = Compose([
                # Resize(320,320),
                # RandomCrop(240, 240),
                RandomRotate90(),
                #RandomGridShuffle(grid=(2, 2), p=0.2),
                HueSaturationValue(p=0.2),
                RandomBrightnessContrast(p=0.2),
                GaussNoise(p=0.2),
                OneOf([
                MotionBlur(),
                GaussianBlur(),
                ImageCompression(quality_lower=65, quality_upper=80),
                ], p=0.2)
            ], p=1)

            img = aug(image=img)['image']
            tfms = data_transform['train']
        else:
            tfms = data_transform['val']
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = tfms(img).unsqueeze(0)
        label = self.labels.iloc[index, 1]
        if label>1:
            label=1
        img = img.squeeze(0)
        # print('face shape:', face.shape)

        return img, label



class MyDataset(Dataset):
    def __init__(self, label_file, root_dir, phase=None):

        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.pahse = phase
        data_transform = {
            "train": transforms.Compose([  # transforms.RandomCrop(240),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.RandomErasing(0.7,scale=(0.05,0.2),ratio=(0.5,2)),
                # transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
            ]),
            "val": transforms.Compose([
                # transforms.CenterCrop(240),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize([0.39, 0.28, 0.27], [0.21, 0.16, 0.15])
            ])
        }
    #  return the number of samples in dataset
    def __len__(self):
        return len(self.labels)

    #  return a sample at the given index
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.pahse == 'train':
            aug = Compose([
                # Resize(320,320),
                # RandomCrop(240, 240),
                RandomRotate90(),
                #RandomGridShuffle(grid=(2, 2), p=0.2),
                HueSaturationValue(p=0.2),
                RandomBrightnessContrast(p=0.2),
                GaussNoise(p=0.2),
                OneOf([
                MotionBlur(),
                GaussianBlur(),
                ImageCompression(quality_lower=65, quality_upper=80),
                ], p=0.2)
            ], p=1)

            img = aug(image=img)['image']
            tfms = data_transform['train']
        else:
            tfms = data_transform['val']
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = tfms(img).unsqueeze(0)
        label = self.labels.iloc[index, 1]
        img = img.squeeze(0)
        # print('face shape:', face.shape)

        return img, label
