"""
이미지 데이터셋 전처리 코드
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
from tqdm import tqdm  # 진행 상황 표시



class BirdDataset(Dataset):
    def __init__(self, image_dir, shape_dir, transform=None, shape_transform=None):
        """
        이미지와 형상 데이터를 로드하는 데이터셋 클래스
        Args:
            image_dir (str): RGB 이미지가 저장된 디렉토리 경로.
            shape_dir (str): 형상 이미지가 저장된 디렉토리 경로.
            transform (callable, optional): RGB 이미지에 적용할 전처리.
            shape_transform (callable, optional): 형상 이미지에 적용할 전처리.
        """
        self.image_dir = image_dir
        self.shape_dir = shape_dir
        self.transform = transform
        self.shape_transform = shape_transform

        # 이미지와 형상 디렉토리에서 클래스 목록 가져오기
        self.classes = sorted(os.listdir(image_dir))

        # 이미지 경로와 클래스 레이블 매핑
        self.image_paths = []
        self.shape_paths = []
        self.labels = []
        for idx, cls in enumerate(self.classes):
            image_class_dir = os.path.join(image_dir, cls)
            shape_class_dir = os.path.join(shape_dir, cls)

            # 해당 클래스의 모든 이미지와 형상 이미지 경로 수집
            image_files = os.listdir(image_class_dir)
            shape_files = os.listdir(shape_class_dir)

            for img_file, shape_file in zip(image_files, shape_files):
                self.image_paths.append(os.path.join(image_class_dir, img_file))
                self.shape_paths.append(os.path.join(shape_class_dir, shape_file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # RGB 이미지 로드
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # 형상 이미지 로드 (흑백이므로 변환 불필요)
        shape_path = self.shape_paths[idx]
        shape = Image.open(shape_path)  # 흑백 이미지를 그대로 사용

        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        if self.shape_transform:
            shape = self.shape_transform(shape)

        label = self.labels[idx]
        return image, shape, label


# 전처리 정의
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

shape_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 흑백 이미지는 [0, 1] 범위로 변환
])

# 데이터셋 초기화
train_dataset = BirdDataset(
    image_dir="lowIBirdMGClassification/resource/images",
    shape_dir="lowIBirdMGClassification/resource/shapes",
    transform=image_transform,
    shape_transform=shape_transform
)

# DataLoader로 배치 단위로 묶기
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# HDF5 파일 경로 설정
hdf5_file_path = "lowIBirdMGClassification/resource/preprocessed/bird_dataset.h5"

# HDF5 파일 생성
with h5py.File(hdf5_file_path, "w") as hdf5_file:
    # 데이터셋 크기 계산
    total_samples = len(train_dataset)
    
    # HDF5 데이터셋 생성 (배열 크기 사전 정의)
    images_dataset = hdf5_file.create_dataset(
        "images", (total_samples, 224, 224, 3), dtype=np.float32
    )
    shapes_dataset = hdf5_file.create_dataset(
        "shapes", (total_samples, 224, 224, 1), dtype=np.float32
    )
    labels_dataset = hdf5_file.create_dataset(
        "labels", (total_samples,), dtype=np.integer
    )
    
    # 데이터 저장
    idx = 0
    for images, shapes, labels in tqdm(train_loader, desc="Saving to HDF5"):
        batch_size = images.size(0)
        images_dataset[idx:idx + batch_size] = images.permute(0, 2, 3, 1).numpy()  # [C, H, W] → [H, W, C]
        shapes_dataset[idx:idx + batch_size] = shapes.permute(0, 2, 3, 1).numpy()  # [C, H, W] → [H, W, C]
        labels_dataset[idx:idx + batch_size] = labels.numpy()
        idx += batch_size

# 데이터 확인
for images, shapes, labels in train_loader:
    print("RGB Images Shape:", images.shape)  # [B, 3, 224, 224]
    print("Shape Images Shape:", shapes.shape)  # [B, 1, 224, 224]
    print("Labels Shape:", labels.shape)  # [B]
    break
