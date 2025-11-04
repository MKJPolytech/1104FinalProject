import os, sys, json, argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model
from keras import layers, Model

#MODEL_PATH = "/workspace/GradProject/saved_model/model_lines_Ver06.keras" 하드코딩된 거
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "saved_model", "model_lines_Ver06.keras")
)
IMAGE_SIZE = (848, 64)  # (W, H)
INPUT_SHAPE = (IMAGE_SIZE[1], IMAGE_SIZE[0], 1)  # (H, W, C) = (64, 848, 1)
TH_HIGH = 0.75
TH_MID  = 0.55

# ----- Otsu 동일 로직 -----
def _otsu_threshold(gray_uint8):
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256))
    total = gray_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0; w_b = 0.0; var_max = 0.0; thresh = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between; thresh = t
    return thresh

def l1_distance(tensors):
    x, y = tensors
    return tf.abs(x - y)

def build_base_cnn(input_shape=INPUT_SHAPE):
    # companion.py 구조와 최대한 동일하게
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    return Model(inp, x, name="base_cnn")

def build_siamese_network_local(input_shape=INPUT_SHAPE):
    base = build_base_cnn(input_shape)
    a = layers.Input(shape=input_shape, name="img_a")
    b = layers.Input(shape=input_shape, name="img_b")
    fa = base(a)
    fb = base(b)
    # 이름 있는 함수 + output_shape 지정
    dist = layers.Lambda(l1_distance, output_shape=(256,), name="l1_distance")([fa, fb])
    out = layers.Dense(1, activation='sigmoid', name="similarity")(dist)
    return Model([a, b], out, name="siamese")

def load_siamese_model_resilient():
    """
    3단계 전략:
    A) 일반 로드 (필요 시 unsafe 허용)
    B) 동일 구조 재생성 후 load_weights()
    C) 마지막 재시도
    """
    # A) 일반 로드
    try:
        try:
            return load_model(MODEL_PATH, compile=False)
        except (ValueError, NotImplementedError) as e:
            # Lambda/unsafe 문제 → 비안전 역직렬화 허용 후 재시도
            if "lambda" in str(e).lower() or "unsafe" in str(e).lower():
                keras.config.enable_unsafe_deserialization()
                return load_model(
                    MODEL_PATH,
                    compile=False,
                    safe_mode=False,
                    custom_objects={"l1_distance": l1_distance},
                )
            raise
    except Exception:
        pass  # Plan B로

    # B) 구조 재생성 → 가중치만 로드
    # 1) companion.py의 원본 빌더가 있으면 그걸 우선 사용
    try:
        from companion import build_siamese_network  # 너의 파일에 있을 가능성 큼
        model = build_siamese_network(input_shape=INPUT_SHAPE)
        model.load_weights(MODEL_PATH)  # 가중치만 로드
        return model
    except Exception:
        # 2) 로컬 임시 아키텍처로 재구성
        model = build_siamese_network_local(INPUT_SHAPE)
        try:
            model.load_weights(MODEL_PATH)
            return model
        except Exception as e:
            # C) 마지막 재시도: SavedModel 디렉터리일 가능성 등
            try:
                return tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects={"l1_distance": l1_distance},
                    safe_mode=False,
                )
            except Exception:
                raise e

def preprocess_image(img_path, image_size=IMAGE_SIZE):
    img = Image.open(img_path).convert("L").resize(image_size, Image.BILINEAR)
    arr = np.array(img)  # (H,W)
    t = _otsu_threshold(arr.astype(np.uint8))
    arr = (arr >= t).astype("float32")
    H, W = image_size[1], image_size[0]
    arr = arr.reshape(H, W, 1).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,1)
    return arr

# ----- 품질 가드: 잉크비율 + 텍스트성(행/열 잉크 분포) -----
def ink_and_textlike(path, white_threshold=240):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    ink = arr < white_threshold
    ink_ratio = float(ink.sum()) / float(arr.size)
    rows = (ink.any(axis=1)).mean()
    cols = (ink.any(axis=0)).mean()
    textlike = float((rows + cols) / 2.0)
    return ink_ratio, textlike

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_A")
    ap.add_argument("image_B")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # 최소 품질 가드
    inkA, txtA = ink_and_textlike(args.image_A)
    inkB, txtB = ink_and_textlike(args.image_B)
    if (inkA < 0.001 or inkB < 0.001) or (txtA < 0.03 or txtB < 0.03):
        print(f"GUARD|msg=입력 품질이 너무 낮아 분석 불가|inkA={inkA}|inkB={inkB}|txtA={txtA}|txtB={txtB}")
        return 2

    # ---- 모델 로드 & 예측 (예외 안전) ----
    try:
        model = load_siamese_model_resilient()
        a = preprocess_image(args.image_A)
        b = preprocess_image(args.image_B)
        y = model.predict([a, b], verbose=0)
        score = float(y[0][0])
    except Exception:
        import traceback
        print("ERROR|예측 단계에서 예외 발생", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ---- 정상 출력 ----
    verdict = ("same" if score >= TH_HIGH else "maybe" if score >= TH_MID else "diff")
    message = {
        "same":  "같은 작성자일 가능성이 높습니다.",
        "maybe": "동일 작성자일 수도 있습니다.",
        "diff":  "다른 작성자일 가능성이 높습니다."
    }[verdict]

    if args.json:
        out = {"ok": True, "cosine_similarity": score, "judgment": message}
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"🔍 유사도 점수: {score:.4f}")
        print(message)
    return 0

if __name__ == "__main__":
    sys.exit(main())
