import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    # ImageNet 평균값 사용
])

# 데이터셋 로드 (고해상도 데이터셋 경로 필요)
train_dataset = datasets.ImageFolder('resource/train/images', transform=transform)
val_dataset = datasets.ImageFolder('resource/val/images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 클래스 이름 매핑
class_names = train_dataset.classes  # ['class_1', 'class_2', ...]

print(class_names)

# GPU 사용할 수 있으면 변경
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_classification_model(_num_classes):
    _model = models.efficientnet_b0(pretrained=True)
    _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, _num_classes)
    return _model

num_classes = len(class_names)
model = build_classification_model(num_classes).to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습 루프
def train_model(_model, _train_loader, _val_loader, epochs=10):
    for epoch in range(epochs):
        _model.train()
        running_loss = 0.0
        for images, labels in tqdm(_train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = _model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(_train_loader)}")

        # Validation
        validate_model(_model, _val_loader, epoch)


def validate_model(_model, _val_loader, epoch):
    _model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in _val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = _model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if not os.path.exists('models'):
        os.makedirs('models')
    path = os.path.join('models', f'model_{epoch + 1}.pth')
    torch.save(_model, path)
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')


# 모델 학습 시작
train_model(model, train_loader, val_loader, epochs=10)

torch.save(model, 'model.pth')
