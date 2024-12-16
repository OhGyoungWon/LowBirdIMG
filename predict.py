import os
import torch
import json
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as functional


with open('class_names.json', 'r', encoding='utf-8') as json_file:
    class_names = json.load(json_file)

# GPU 사용할 수 있으면 변경
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth', weights_only=False)
model.eval()

def predict_bird_species(image_path, model, class_names):
    _image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    sr_tensor = transform(_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(sr_tensor)
        probabilities = functional.softmax(outputs, dim=1).squeeze()

    top3_indices = torch.topk(probabilities, k=3).indices.tolist()
    top3_probabilities = torch.topk(probabilities, k=3).values.tolist()
    top3_classes = [(class_names[idx], prob * 100, idx + 1) for idx, prob, idx in zip(top3_indices, top3_probabilities, top3_indices)]

    print("예측 결과:")
    for rank, (class_name, prob, idx) in enumerate(top3_classes, 1):
        print(f"{rank}. {class_name}: {prob:.1f}%")

    result = {
        "top3_classes": top3_classes
    }
    return result


# 테스트 예측
lr_image_path = 'resource/test/images/sample_bird.jpg'
result = predict_bird_species(lr_image_path, model, class_names)
print(f"Predicted Bird Species: {result}")
