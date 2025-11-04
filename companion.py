# companion.py (tf.data 파이프라인 적용판)

import os
import random
from glob import glob
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import ModelCheckpoint

import random as pyrandom
np.random.seed(42); pyrandom.seed(42); tf.random.set_seed(42)

AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------
# 데이터 로딩 (경로만 수집)
# ------------------------------
def load_images_by_id(base_path):
    image_by_id = {}
    for subdir, _, _ in os.walk(base_path):
        image_paths = glob(os.path.join(subdir, "*.png"))
        if len(image_paths) >= 2:
            # 필자 ID는 가장 안쪽 폴더 이름 (e.g., a01-000u)
            id_ = os.path.basename(subdir)
            image_by_id[id_] = image_paths
            print(f"📁 {id_} → {len(image_paths)}장 이미지")
    print(f"📊 총 필자 수: {len(image_by_id)}")
    return image_by_id

# ------------------------------
# 전처리 (PIL + Otsu 로직 유지)
# ------------------------------
def _otsu_threshold(gray_uint8):
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256))
    total = gray_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, var_max, thresh = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            thresh = t
    return thresh

def preprocess(path, size=(848, 64), to_binary=True):
    # PIL은 (W,H) 기준으로 resize하므로 size=(W,H)
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.array(img)  # (H, W)

    if to_binary:
        t = _otsu_threshold(arr.astype(np.uint8))
        arr = (arr >= t).astype("float32")    # 0/1
    else:
        arr = arr.astype("float32") / 255.0

    arr = arr[..., None]  # (H, W, 1)
    return arr

# --- tf.data용 전처리 (파일 경로 → 텐서) ---
def _apply_otsu_np(gray_f32_0_1):
    # gray_f32_0_1: [H,W] float32, 0~1
    arr_u8 = (gray_f32_0_1 * 255.0).astype(np.uint8)
    t = _otsu_threshold(arr_u8)
    bin_img = (arr_u8 >= t).astype(np.float32)
    return bin_img  # [H,W] float32

def preprocess_tf(file_path, image_size=(848, 64), to_binary=True):
    # 1) 파일 읽기 & 디코드
    file_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)  # [H,W,1], uint8
    img = tf.image.convert_image_dtype(img, tf.float32)                        # [H,W,1], 0~1

    # 2) 리사이즈 (W,H 주의: image_size=(W,H))
    W, H = image_size
    img = tf.image.resize(img, size=(H, W), method=tf.image.ResizeMethod.BILINEAR)  # [H,W,1]

    # 3) (옵션) Otsu 이진화 (numpy_function 사용)
    if to_binary:
        gray = tf.squeeze(img, axis=-1)   # [H,W]
        binimg = tf.numpy_function(_apply_otsu_np, [gray], Tout=tf.float32)
        binimg.set_shape([H, W])          # shape 복구
        img = binimg[..., tf.newaxis]     # [H,W,1]
    # (to_binary=False이면 0~1 float 그대로)
    return img  # [H,W,1] float32

# ------------------------------
# tf.data: 경로 제너레이터 & 파이프라인
# ------------------------------
def iter_same_then_diff_paths(image_by_id, max_pairs_per_id=5):
    """
    무한 제너레이터:
      - 각 writer(id)에서 인접 이미지로 same 쌍 생성(최대 max_pairs_per_id)
      - same 하나당 diff 하나 즉석 생성
    산출: (path1, path2, label)  // label in {0,1}
    """
    ids = list(image_by_id.keys())
    rng = random.Random(42)
    while True:
        rng.shuffle(ids)
        for id_ in ids:
            paths = image_by_id[id_]
            if len(paths) < 2:
                continue
            limit = min(len(paths) - 1, max_pairs_per_id)
            for i in range(limit):
                p1, p2 = paths[i], paths[i + 1]
                # same
                yield (p1, p2, 1)
                # diff (서로 다른 필자)
                id1, id2 = rng.sample(ids, 2)
                q1 = rng.choice(image_by_id[id1])
                q2 = rng.choice(image_by_id[id2])
                yield (q1, q2, 0)

def build_pair_dataset(image_by_id,
                       batch_size=32,
                       image_size=(848, 64),
                       to_binary=True,
                       max_pairs_per_id=5,
                       buffer_size=8192):
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),  # path1
        tf.TensorSpec(shape=(), dtype=tf.string),  # path2
        tf.TensorSpec(shape=(), dtype=tf.int32),   # label
    )
    gen = lambda: iter_same_then_diff_paths(image_by_id, max_pairs_per_id=max_pairs_per_id)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def _load_map(p1, p2, y):
        x1 = preprocess_tf(p1, image_size=image_size, to_binary=to_binary)
        x2 = preprocess_tf(p2, image_size=image_size, to_binary=to_binary)
        y  = tf.cast(y, tf.float32)
        return (x1, x2), y

    ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
    ds = ds.map(_load_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def split_ids(image_by_id, val_ratio=0.1, seed=42):
    ids = list(image_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = {k: v for k, v in image_by_id.items() if k not in val_ids}
    val   = {k: v for k, v in image_by_id.items() if k in val_ids}
    return train, val

# ------------------------------
# CNN / Siamese 모델 (기존 유지)
# ------------------------------
def build_base_cnn(input_shape=(64, 848, 1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    return Model(inp, x)

def build_siamese_network():
    input_shape = (64, 848, 1)
    base_cnn = build_base_cnn(input_shape)
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)
    feat1 = base_cnn(input1)
    feat2 = base_cnn(input2)
    l1 = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([feat1, feat2])
    out = layers.Dense(1, activation='sigmoid')(l1)
    model = Model([input1, input2], out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ------------------------------
# 학습 (tf.data 사용)
# ------------------------------
def train_with_tfdata(base_path,
                      model_name="model_lines_Ver06.keras",
                      batch_size=32,
                      image_size=(848, 64),
                      to_binary=True,
                      max_pairs_per_id=5,
                      epochs=7):
    from keras.callbacks import EarlyStopping

    # 1) 경로 로드 & ID 스플릿
    image_by_id = load_images_by_id(base_path)
    train_ids, val_ids = split_ids(image_by_id, val_ratio=0.1)

    # 2) Dataset 구성
    train_ds = build_pair_dataset(
        train_ids, batch_size=batch_size, image_size=image_size,
        to_binary=to_binary, max_pairs_per_id=max_pairs_per_id
    )
    val_ds = build_pair_dataset(
        val_ids, batch_size=batch_size, image_size=image_size,
        to_binary=to_binary, max_pairs_per_id=max_pairs_per_id
    )

    # steps_per_epoch/validation_steps 대략치 (ID, 쌍 수 기반 추정)
    approx_pairs_per_cycle = max_pairs_per_id * 2 * max(1, len(train_ids))  # same+diff
    steps_per_epoch = max(200, approx_pairs_per_cycle // batch_size)
    val_steps = max(50, max_pairs_per_id * 2 * max(1, len(val_ids)) // batch_size)

    print(f"[train] steps_per_epoch={steps_per_epoch}, val_steps={val_steps}")

    # 3) 모델
    model = build_siamese_network()

    # 4) 저장 경로 & 콜백
    save_dir = "/workspace/GradProject/saved_model"
    os.makedirs(save_dir, exist_ok=True)
    full_model_path = os.path.join(save_dir, model_name)

    checkpoint_cb = ModelCheckpoint(
        full_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        mode="max",
        restore_best_weights=True
    )

    # 5) 학습
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_cb],
        verbose=1
    )

    # 6) 학습 곡선
    plot_training_history(history)
    return model

# ------------------------------
# 비교 / 시각화 (기존 유지)
# ------------------------------
def evaluate_model_with_train(model, x1_train, x2_train, y_train, x1_val, x2_val, y_val):
    preds_val = model.predict([x1_val, x2_val])
    preds_val_class = (preds_val > 0.5).astype("int32").flatten()
    acc_val = accuracy_score(y_val, preds_val_class)

    preds_train = model.predict([x1_train, x2_train])
    preds_train_class = (preds_train > 0.5).astype("int32").flatten()
    acc_train = accuracy_score(y_train, preds_train_class)

    print(f"\n✅ Training Accuracy: {acc_train:.4f}")
    print(f"✅ Validation Accuracy: {acc_val:.4f}")

    cm = confusion_matrix(y_val, preds_val_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Validation Confusion Matrix")
    plt.show()

def compare_two_images(model_path, img_path1, img_path2, image_size=(848, 64)):
    def preprocess_image(path):
        img = Image.open(path).convert("L").resize(image_size, Image.BILINEAR)
        arr = np.array(img)
        # 학습과 동일하게 Otsu
        t = _otsu_threshold(arr.astype(np.uint8))
        arr = (arr >= t).astype("float32")
        arr = arr[..., None]                                  # (H, W, 1)
        arr = arr.reshape(1, image_size[1], image_size[0], 1) # (1, H, W, 1)
        return arr
    img1 = preprocess_image(img_path1)
    img2 = preprocess_image(img_path2)
    model = tf.keras.models.load_model(model_path, compile=False)
    pred = model.predict([img1, img2])[0][0]
    print(f"\n🔍 Similarity Score: {pred:.4f}")
    print("✅ Same Writer" if pred > 0.5 else "❌ Different Writers")
    return pred

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.show()

# ------------------------------
# 엔트리 포인트
# ------------------------------
if __name__ == "__main__":
    base_lines = r"/workspace/GradProject/IAM Database/IAM Handwriting Database Sets/lines"
    # tf.data 파이프라인 기반 학습
    train_with_tfdata(
        base_lines,
        model_name="model_lines_Ver06.keras",
        batch_size=32,
        image_size=(848, 64),
        to_binary=True,
        max_pairs_per_id=5,
        epochs=7
    )
# 실행 권장:  python -u companion.py
