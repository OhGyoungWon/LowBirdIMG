import os
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. 데이터셋 경로 설정
original_rgb_dir = 'lowIBirdMGClassification/resource/origin/images'  # 원본 RGB 데이터 경로
original_mask_dir = 'lowIBirdMGClassification/resource/origin/shapes'  # 흑백 형상 데이터 경로
train_rgb_dir = 'lowIBirdMGClassification/resource/train/images'
train_mask_dir = 'lowIBirdMGClassification/resource/train/shapes'
val_rgb_dir = 'lowIBirdMGClassification/resource/val/images'
val_mask_dir = 'lowIBirdMGClassification/resource/val/shapes'

# 2. 데이터 분할 함수
def split_dual_data(original_rgb_dir, original_mask_dir, train_rgb_dir, val_rgb_dir, train_mask_dir, val_mask_dir, test_size=0.2):
    """
    데이터를 학습 및 검증 데이터로 분할하여 디렉토리에 저장.
    기존 분할된 데이터가 있으면 작업을 건너뜀.
    """
    # 분류 상태 파일
    done_flag = os.path.join(train_rgb_dir, '.done')
    if os.path.exists(done_flag):
        print("데이터가 이미 분류되어 있습니다. 작업을 건너뜁니다.")
        return

    print("데이터를 분류 중입니다...")
    classes = os.listdir(original_rgb_dir)
    for cls in classes:
        rgb_cls_path = os.path.join(original_rgb_dir, cls)
        mask_cls_path = os.path.join(original_mask_dir, cls)

        rgb_images = os.listdir(rgb_cls_path)
        mask_images = os.listdir(mask_cls_path)

        # train:val = 80:20 비율로 나눔
        train_rgb, val_rgb, train_mask, val_mask = train_test_split(
            rgb_images, mask_images, test_size=test_size, random_state=42
        )

        # RGB 데이터 저장
        train_rgb_cls_path = os.path.join(train_rgb_dir, cls)
        val_rgb_cls_path = os.path.join(val_rgb_dir, cls)
        os.makedirs(train_rgb_cls_path, exist_ok=True)
        os.makedirs(val_rgb_cls_path, exist_ok=True)
        for img in train_rgb:
            shutil.copy(os.path.join(rgb_cls_path, img), os.path.join(train_rgb_cls_path, img))
        for img in val_rgb:
            shutil.copy(os.path.join(rgb_cls_path, img), os.path.join(val_rgb_cls_path, img))

        # 마스크 데이터 저장
        train_mask_cls_path = os.path.join(train_mask_dir, cls)
        val_mask_cls_path = os.path.join(val_mask_dir, cls)
        os.makedirs(train_mask_cls_path, exist_ok=True)
        os.makedirs(val_mask_cls_path, exist_ok=True)
        for img in train_mask:
            shutil.copy(os.path.join(mask_cls_path, img), os.path.join(train_mask_cls_path, img))
        for img in val_mask:
            shutil.copy(os.path.join(mask_cls_path, img), os.path.join(val_mask_cls_path, img))

    # 분류 완료 상태 표시
    with open(done_flag, 'w') as f:
        f.write("Data split completed.")
    print("데이터 분류가 완료되었습니다.")



# 데이터 분할 실행
split_dual_data(original_rgb_dir, original_mask_dir, train_rgb_dir, val_rgb_dir, train_mask_dir, val_mask_dir)


# 3. 데이터 로드 (ImageDataGenerator)
def load_dual_data(rgb_dir, mask_dir, target_size=(300, 300), batch_size=32):
    rgb_datagen = ImageDataGenerator(rescale=1.0/255)
    mask_datagen = ImageDataGenerator(rescale=1.0/255)

    rgb_generator = rgb_datagen.flow_from_directory(
        rgb_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb'
    )

    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    def generator():
        while True:
            rgb_batch, labels = next(rgb_generator)
            mask_batch, _ = next(mask_generator)
            yield ([rgb_batch, mask_batch], labels)

    # 클래스 인덱스를 추출
    class_indices = rgb_generator.class_indices

    # Output signature를 명시적으로 정의
    output_signature = (
        (
            tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),  # RGB 이미지
            tf.TensorSpec(shape=(None, *target_size, 1), dtype=tf.float32)   # 마스크 이미지
        ),
        tf.TensorSpec(shape=(None, len(class_indices)), dtype=tf.float32)  # 클래스 레이블
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset, class_indices



# 수정된 데이터 로더 사용
train_generator, train_class_indices = load_dual_data(train_rgb_dir, train_mask_dir)
val_generator, val_class_indices = load_dual_data(val_rgb_dir, val_mask_dir)

# 1. 데이터 클래스 확인


# 클래스 이름 저장
class_names = list(train_class_indices.keys())
print("Train class names:", class_names)

# 4. EfficientNet 및 추가 CNN 모델 정의
rgb_input = Input(shape=(300, 300, 3), name='rgb_input')
mask_input = Input(shape=(300, 300, 1), name='mask_input')

# RGB 이미지 처리 (EfficientNet)
base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=rgb_input)
x_rgb = base_model.output
x_rgb = GlobalAveragePooling2D()(x_rgb)

# 흑백 형상 이미지 처리 (간단한 CNN)
x_mask = Conv2D(32, (3, 3), activation='relu', padding='same')(mask_input)
x_mask = GlobalAveragePooling2D()(x_mask)

# 두 입력 병합 및 최종 분류
merged = Concatenate()([x_rgb, x_mask])
x = Dense(1024, activation='relu')(merged)
predictions = Dense(len(train_class_indices), activation='softmax')(x)

model = Model(inputs=[rgb_input, mask_input], outputs=predictions)

# 5. 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 모델 학습
epochs = 10
steps_per_epoch = len(os.listdir(train_rgb_dir)) * 200 // 32  # 클래스당 200장 기준
validation_steps = len(os.listdir(val_rgb_dir)) * 200 // 32

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# 7. 모델 저장
model.save('lowIBirdMGClassification/saveModel')

# 8. 예측 함수 (상위 3개 클래스 및 최종 분류)
def predict_image_with_top3(rgb_image_path, mask_image_path=None):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # RGB 이미지 로드 및 전처리
    rgb_img = load_img(rgb_image_path, target_size=(300, 300))
    rgb_array = img_to_array(rgb_img) / 255.0  # 정규화
    rgb_array = tf.expand_dims(rgb_array, axis=0)  # 배치 차원 추가

    # 흑백 형상 이미지 로드 및 전처리
    if mask_image_path:
        mask_img = load_img(mask_image_path, target_size=(300, 300), color_mode='grayscale')
        mask_array = img_to_array(mask_img) / 255.0
        mask_array = tf.expand_dims(mask_array, axis=0)
    else:
        # 흑백 형상 이미지가 없으면 빈 배열 사용
        mask_array = np.zeros((1, 300, 300, 1))

    # 모델 예측
    predictions = model.predict([rgb_array, mask_array])[0]  # 첫 번째 배치의 결과
    top3_indices = np.argsort(predictions)[-3:][::-1]  # 확률 상위 3개 인덱스 (내림차순)
    top3_classes = [(class_names[i], predictions[i] * 100) for i in top3_indices]  # 클래스명과 확률 계산

    # 최종 예측 클래스와 확률
    predicted_class = class_names[top3_indices[0]]
    confidence = predictions[top3_indices[0]] * 100

    # 출력 포맷팅
    print(f"입력 이미지: {os.path.basename(rgb_image_path)}")
    print(f"예측된 상위 3개 클래스:")
    for rank, (class_name, prob) in enumerate(top3_classes, 1):
        print(f"{rank}. {class_name}: {prob:.1f}%")
    print(f"\n해당 이미지는 {confidence:.1f}% 확률로 \"{predicted_class}\" 입니다.\n")

    return top3_classes, predicted_class, confidence

# 테스트 예측
test_rgb_path = 'lowIBirdMGClassification/resource/test_Img.jpg'
test_mask_path = None  # 테스트에서 흑백 형상 이미지를 제공하지 않는 경우

# 예측 실행
top3_classes, predicted_class, confidence = predict_image_with_top3(test_rgb_path, test_mask_path)

