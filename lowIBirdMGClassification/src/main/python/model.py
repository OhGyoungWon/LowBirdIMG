from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
from PIL import Image
import numpy as np
import torch

def upscale_image_with_realesrgan(lr_image_path):
    """
    Real-ESRGAN을 사용하여 저해상도 이미지를 고해상도로 변환합니다.
    """
    # Real-ESRGAN 모델 아키텍처 정의
    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Real-ESRGAN 모델 초기화
    model = RealESRGANer(
        scale=4,
        model_path='lowIBirdMGClassification/src/main/resource/trained_real_ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
        model=model_arch,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,  # CPU 환경에서 False로 설정
        device=torch.device("cpu")  # CPU에서 실행
    )

    # 저해상도 이미지 로드 (PIL -> Numpy 변환)
    pil_image = Image.open(lr_image_path).convert("RGB")
    lr_image = np.array(pil_image)  # PIL 이미지를 numpy 배열로 변환

    # 업스케일링 수행
    sr_image, _ = model.enhance(lr_image, outscale=4)  # outscale로 업스케일 배율 설정 가능

    # 결과를 Numpy -> PIL 이미지로 변환
    sr_image = Image.fromarray(cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB))
    return sr_image




import torch
import torch.nn as nn
from torchvision import models

def build_classification_model(num_classes):
    """
    EfficientNet B0 기반 분류 모델 생성.
    """
    model = models.efficientnet_b0(pretrained=True)
    # 출력 레이어 수정 (클래스 개수로 조정)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet 입력 크기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 평균 및 표준편차 사용
])

# 데이터셋 경로
dataset_path = "lowIBirdMGClassification/src/main/resource/origin/images"

# 데이터셋 로드
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# 클래스 이름 확인
class_names = dataset.classes  # ['001.Black_footed_Albatross', '002.Laysan_Albatross', ...]

# 학습 및 검증 데이터셋 분리
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

# SubsetRandomSampler로 학습 및 검증 데이터로 나누기
from torch.utils.data.sampler import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# 클래스 확인
print(f"Classes: {class_names}")
print(f"Number of Classes: {len(class_names)}")



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# CPU 사용 설정
device = torch.device("cpu")

# EfficientNet B0 모델 초기화
num_classes = len(class_names)
model = models.efficientnet_b0(pretrained=True)

# 분류기 수정
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프 정의
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 학습 단계
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # 검증 단계
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')


def save_model(model, path):
    """
    학습된 모델 가중치를 저장합니다.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    저장된 가중치를 로드하여 모델을 초기화합니다.
    """
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()  # 평가 모드로 설정
    print(f"Model loaded from {path}")
    return model

# 학습된 모델 저장 경로
model_save_path = "lowIBirdMGClassification/saveModel/model.pth"

# 모델 학습 후 저장
#train_model(model, train_loader, val_loader, epochs=10)
save_model(model, model_save_path)

# 저장된 모델 로드
loaded_model = build_classification_model(num_classes).to(device)  # 모델 초기화
loaded_model = load_model(loaded_model, model_save_path)


import torch.nn.functional as F

def predict_bird_species_with_realesrgan(lr_image_path, model, class_names):
    """
    저해상도 이미지를 Real-ESRGAN으로 업스케일링한 후 분류 모델로 예측합니다.
    """
    # 1. Real-ESRGAN 업스케일링
    sr_image = upscale_image_with_realesrgan(lr_image_path)
    
    # 2. 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet 입력 크기
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    sr_tensor = transform(sr_image).unsqueeze(0)  # 배치 차원 추가
    
    # 3. 분류 모델 예측
    model.eval()
    with torch.no_grad():
        outputs = model(sr_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = probabilities.argmax(dim=1).item()
    
    # 4. 클래스 이름 반환
    return class_names[predicted_class_idx]

# 테스트 실행
lr_image_path = 'lowIBirdMGClassification/src/main/resource/test_03.jpg'
predicted_class = predict_bird_species_with_realesrgan(lr_image_path, loaded_model, class_names)

print(f"Predicted Bird Species: {predicted_class}")
