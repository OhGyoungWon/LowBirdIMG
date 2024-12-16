import keras
import os
import time
from PIL import Image
import tensorflow as tf


# 1. 데이터셋 경로 설정
IMAGE_PATH = os.path.join('resource', 'test', 'images', 'sample_bird.jpg')
SAVED_MODEL_PATH = os.path.join('upscale_model')


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


hr_image = preprocess_image(IMAGE_PATH)
model = keras.models.load_model(SAVED_MODEL_PATH)
fake_image = model(hr_image)
save_image(tf.squeeze(fake_image), filename="Super Resolution")