import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from albumentations import Compose, RandomCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2


# Custom Dataset Class for loading high-resolution images  고해상도 이미지를 로드하고 전처리를 수행하는 사용자 정의 데이터셋
class HighResImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        # Load image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformations
        if self.transform:
            image = self.transform(image=image)['image']

        return image


# Transformation pipeline for data augmentation and normalization
def get_transforms(image_size):
    return Compose([
        # 이미지를 고정된 크기로 조정
        transforms.Resize((image_size, image_size)),

        # Data augmentation techniques
        RandomCrop(image_size - 50, image_size - 50, p=0.5),  # 랜덤한 크기 조정 및 수평 반전을 사용해 데이터를 증강
        HorizontalFlip(p=0.5),  # Horizontally flip with 50% chance
        RandomBrightnessContrast(p=0.5),  # 이미지의 밝기 및 대조를 랜덤하게 조정하여 다양한 조명 조건에 대응

        # 이미지를 텐서로 변환하고, torch.Tensor로 모델에 입력할 수 있도록 함
        ToTensorV2(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 이미지를 정규화하여 훈련 성능을 높임
    ])


# Image directory
image_dir = 'path/to/high_resolution_images'

# Define image size and batch size
image_size = 512  # For high-resolution processing
batch_size = 16

# Create Dataset and DataLoader
dataset = HighResImageDataset(image_dir=image_dir, transform=get_transforms(image_size))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Iterate over DataLoader
for batch_idx, images in enumerate(dataloader):
    print(f"Batch {batch_idx + 1} - Image tensor shape: {images.shape}")
    # Here you can use the images for training a model
