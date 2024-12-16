import os
import json
from celery import Celery
import tensorflow as tf
import shutil
import torch
import os
import json
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as functional
import keras

upscale_model = None
device = None
predict_model = None

with open('class_names.json', 'r', encoding='utf-8') as json_file:
    class_names = json.load(json_file)

# Redis
celery_app = Celery('myapp', broker='redis://localhost:6379/0')

def predict_image(model, rgb_image_path, _device):
    _image = Image.open(rgb_image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    sr_tensor = transform(_image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(sr_tensor)
        probabilities = functional.softmax(outputs, dim=1).squeeze()

    top3_indices = torch.topk(probabilities, k=3).indices.tolist()
    top3_probabilities = torch.topk(probabilities, k=3).values.tolist()
    top3_classes = [(class_names[idx], prob * 100, idx + 1) for idx, prob, idx in
                    zip(top3_indices, top3_probabilities, top3_indices)]

    print("예측 결과:")
    for rank, (class_name, prob, idx) in enumerate(top3_classes, 1):
        print(f"{rank}. {class_name}: {prob:.1f}%")

    result = {
        "top3_classes": top3_classes
    }
    return result


def is_image_smaller_than_224px(file_path):
    try:
        # 이미지 열기
        with Image.open(file_path) as img:
            width, height = img.size
            if width < 224 or height < 224:
                return True
            else:
                return False
    except Exception as e:
        print(f"Error while processing the image: {e}")
        return False

def preprocess_image(image_path):
    file_io = tf.io.read_file(image_path)
    _hr_image = tf.image.decode_image(file_io)
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if _hr_image.shape[-1] == 4:
        _hr_image = _hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(_hr_image.shape[:-1]) // 4) * 4
    _hr_image = tf.image.crop_to_bounding_box(_hr_image, 0, 0, hr_size[0], hr_size[1])
    _hr_image = tf.cast(_hr_image, tf.float32)
    return tf.expand_dims(_hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

def upscale_task(model, session):
    img_path = f'../data/temp_image/{session}.jpg'
    if not os.path.isfile(img_path):
        img_path = f'../data/temp_image/{session}.png'
    if not os.path.isfile(img_path):
        img_path = f'../data/temp_image/{session}.jpeg'

    if is_image_smaller_than_224px(img_path):
        hr_image = preprocess_image(img_path)
        fake_image = model(hr_image)
        if isinstance(fake_image, dict) and 'rrdb_net' in fake_image:
            fake_image_tensor = fake_image['rrdb_net']
        else:
            raise ValueError("Unexpected output format from upscale_model. Expected a dict with key 'rrdb_net'.")
        squeezed_image = tf.squeeze(fake_image_tensor)
        save_image(squeezed_image, filename=f'../data/upscale_image/{session}')
    else:
        shutil.copy(img_path, f'../data/upscale_image/{session}.jpg')

@celery_app.task
def predict_task(session):
    global upscale_model
    global device
    global predict_model
    if upscale_model is None:
        upscale_model = keras.layers.TFSMLayer("upscale_model", call_endpoint="serving_default")
    if device is None:
        device = torch.device("mps")
    if predict_model is None:
        predict_model = torch.load('predict_model.pth', map_location=device, weights_only=False)

    print(f"upscale: {session}")
    upscale_task(upscale_model, session)

    # 이미지 경로 설정
    image_path = f'../data/upscale_image/{session}.jpg'

    # 예측 실행
    print(f"predict: {session}")
    result = predict_image(predict_model, image_path, device)
    print(f"predict done: {session}")

    output_path = f'../data/result/{session}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 폴더 생성
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)  # JSON 파일로 저장

# pip install celery
# pip install gevent
# pip install redis
# brew install redis
# brew services start redis
# sudo celery multi start 2 -A celery_app --concurrency=8 --logfile=shell.log --pool=gevent
# sudo celery multi stop 6