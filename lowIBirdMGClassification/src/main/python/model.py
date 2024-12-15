from realesrgan import RealESRGANer
from PIL import Image

def upscale_image_with_realesrgan(lr_image_path):
    """
    Real-ESRGAN을 사용하여 저해상도 이미지를 고해상도로 변환합니다.
    """
    # Real-ESRGAN 모델 초기화 (x4 업스케일링)
    model = RealESRGANer(torch.device("cpu"), scale=4)
    model.load_weights('lowIBirdMGClassification/src/main/resource/trained_real_ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth')  # 사전 학습된 가중치
    
    # 저해상도 이미지 로드
    lr_image = Image.open(lr_image_path).convert("RGB")
    
    # 업스케일링 수행
    sr_image = model.predict(lr_image)
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

# 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 평균값 사용
])

# 데이터셋 로드 (고해상도 데이터셋 경로 필요)
train_dataset = datasets.ImageFolder('lowIBirdMGClassification/src/main/resource/train/images', transform=transform)
val_dataset = datasets.ImageFolder('lowIBirdMGClassification/src/main/resource/val/images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 클래스 이름 매핑
class_names = train_dataset.classes  # ['class_1', 'class_2', ...]


import torch.optim as optim
from tqdm import tqdm

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화
num_classes = len(class_names)
model = build_classification_model(num_classes).to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Validation
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

# 모델 학습 시작
train_model(model, train_loader, val_loader, epochs=10)

# 학습된 모델 저장 경로
model_save_path = "lowIBirdMGClassification/saveModel/bird_classifier.pth"

# 모델 학습 후 저장
train_model(model, train_loader, val_loader, epochs=10)
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
lr_image_path = 'lowIBirdMGClassification/src/main/resource/origin/images/001.Black_footed_Albatross/test_image.jpg'
predicted_class = predict_bird_species_with_realesrgan(lr_image_path, model, class_names)

print(f"Predicted Bird Species: {predicted_class}")


import torch.nn.functional as F

def predict_bird_species(lr_image_path, model, class_names):
    """
    저해상도 이미지를 입력받아 가장 높은 확률의 클래스를 출력합니다.
    """
    # 업스케일링
    
    sr_image = upscale_image_with_realesrgan(lr_image_path)
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    sr_tensor = transform(sr_image).unsqueeze(0).to(device)
    
    # 모델 예측
    model.eval()
    with torch.no_grad():
        outputs = model(sr_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = probabilities.argmax(dim=1).item()
    
    # 클래스 이름 반환
    return class_names[predicted_class_idx]

# 테스트 예측
lr_image_path = 'lowIBirdMGClassification/src/main/resource/origin/images/001.Black_footed_Albatross/test_image.jpg'
predicted_class = predict_bird_species(lr_image_path, model, class_names)

print(f"Predicted Bird Species: {predicted_class}")
