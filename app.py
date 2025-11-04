import sys, argparse, json
import os, traceback, subprocess, base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===== 업로드 폴더 지정 (pycodes/uploads) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "saved_model", "model_lines_Ver06.keras")
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # 폴더 없으면 자동 생성
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 환경변수로 상세 에러를 클라이언트에 노출할지 결정 (기본 True; 필요시 False로)
DEBUG_EXPOSE_ERRORS = os.environ.get("FH_DEBUG", "1") == "1"

# 모델 관련 상수는 여기선 직접 사용하지 않음(verify_siamese.py에서 사용)
# MODEL_PATH = "/workspace/GradProject/saved_model/model_lines_Ver06.keras"
IMAGE_SIZE = (848, 64)  # (W, H)
TH_HIGH = 0.75
TH_MID  = 0.55

def _ensure_uploads_dir():
    """uploads 디렉터리 보장 후 경로 반환"""
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    return app.config["UPLOAD_FOLDER"]

def _save_dataurl_to_png(data_url: str, out_path: str):
    """
    data_url: "data:image/png;base64,...." 또는 그냥 base64 본문
    """
    if "," in data_url and data_url.strip().lower().startswith("data:"):
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))

# ----- (참고) 품질 가드에서 쓰던 간단 측정 유틸, 서버에서는 사용 안 해도 됨 -----
def _otsu_threshold(gray_uint8):
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256))
    total = gray_uint8.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0; w_b = 0.0; var_max = 0.0; thresh = 0
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
            var_max = var_between; thresh = t
    return thresh

def tight_crop_png(path, white_threshold=240, pad=4, min_size=(16, 16)):
    """
    path의 이미지를 열어 흰 배경 기준으로 타이트 크롭 후, 같은 경로에 덮어씁니다.
    - white_threshold: 이 값보다 작은 픽셀을 '잉크(글씨)'로 간주
    - pad: 크롭 박스에 사방 여유 픽셀
    - min_size: 너무 작게 잘리는 것 방지용 최소 크기 (W,H)
    """
    img = Image.open(path).convert("L")          # 그레이스케일
    arr = np.array(img)

    # '잉크' 마스크 (True = 글씨)
    ink = arr < white_threshold
    if not ink.any():
        # 글씨가 전혀 없으면 그대로 반환
        return

    ys, xs = np.where(ink)                       # 잉크가 있는 좌표들의 y,x
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    # 패딩 및 경계 보정
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(arr.shape[0] - 1, bottom + pad)
    right = min(arr.shape[1] - 1, right + pad)

    # 최소 크기 보장
    h = bottom - top + 1
    w = right - left + 1
    min_w, min_h = min_size
    if w < min_w:
        extra = (min_w - w) // 2 + 1
        left = max(0, left - extra)
        right = min(arr.shape[1] - 1, right + extra)
    if h < min_h:
        extra = (min_h - h) // 2 + 1
        top = max(0, top - extra)
        bottom = min(arr.shape[0] - 1, bottom + extra)

    cropped = img.crop((left, top, right + 1, bottom + 1))
    # 배경을 흰색으로 유지하고 싶다면 아래처럼 RGB/알파 없이 저장
    cropped.save(path)  # 같은 경로에 덮어쓰기

@app.post("/verify")
def verify():
    try:
        data = request.get_json(force=True) or {}
        imgA_b64 = data.get("imageA")
        imgB_b64 = data.get("imageB")
        if not imgA_b64 or not imgB_b64:
            return jsonify({"ok": False, "reason": "BAD_REQUEST", "message": "imageA/B 누락"}), 400

        # 1) A/B 저장 (uploads 경로)
        updir = _ensure_uploads_dir()
        
        def _unique_timestamp_path(updir: str, tag: str) -> str:
            """
            yyyy_mm_dd_hh_mm_ss_{tag}.png 형식으로 저장.
            같은 초(second) 안에 여러 장 저장될 때는 _01, _02...로 중복 회피.
            """
            base = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            path = os.path.join(updir, f"{base}_{tag}.png")
            if not os.path.exists(path):
                return path
            i = 1
            while True:
                path_i = os.path.join(updir, f"{base}_{tag}_{i:02d}.png")
                if not os.path.exists(path_i):
                    return path_i
                i += 1

        pathA = _unique_timestamp_path(updir, "A")
        pathB = _unique_timestamp_path(updir, "B")

        _save_dataurl_to_png(imgA_b64, pathA)
        _save_dataurl_to_png(imgB_b64, pathB)

        # ✅ 자동 좌우+상하 크롭
        tight_crop_png(pathA)
        tight_crop_png(pathB)

        # (필요하면 여기서 자동 크롭 로직 추가 가능 — 현재는 원본 그대로 전달)

        # 2) 외부 스크립트 실행 (텍스트 출력 파싱 모드)
        # cmd = ["python3", "verify_siamese.py", pathA, pathB]  # --json 없이 텍스트 파싱 유지, python3 대신 밑의 sys.executable 사용
        cmd = [sys.executable, "verify_siamese.py", pathA, pathB]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rc, stdout, stderr = proc.returncode, proc.stdout.strip(), proc.stderr.strip()

        # 디버그: 서버 콘솔에도 남겨두기
        print(f"[verify] rc={rc}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}")

        # a) 품질 가드: 종료코드 2 + "GUARD|" 프리픽스
        if rc == 2 and stdout.startswith("GUARD|"):
            msg = "입력 품질이 낮아 분석 불가"
            if "msg=" in stdout:
                msg = stdout.split("msg=", 1)[1].split("|", 1)[0]
            payload = {
                "ok": False,
                "reason": "QUALITY_GUARD",
                "message": msg,
                "saved_paths": {"imageA": pathA, "imageB": pathB}
            }
            if DEBUG_EXPOSE_ERRORS:
                payload["raw_stdout"] = stdout
                payload["raw_stderr"] = stderr
            return jsonify(payload), 200

        # b) 정상: 한국어 출력 파싱
        if rc == 0:
            score = None
            judgment = ""
            for line in stdout.splitlines():
                if "유사도 점수" in line:
                    import re
                    m = re.search(r"([0-9]*\.?[0-9]+)", line)
                    if m:
                        score = float(m.group(1))
                elif line.strip():
                    judgment = judgment or line.strip()

            if score is None:
                payload = {
                    "ok": False,
                    "reason": "PARSE_FAIL",
                    "message": "분석 결과 파싱 실패",
                    "saved_paths": {"imageA": pathA, "imageB": pathB}
                }
                if DEBUG_EXPOSE_ERRORS:
                    payload["raw_stdout"] = stdout
                    payload["raw_stderr"] = stderr
                return jsonify(payload), 500

            return jsonify({
                "ok": True,
                "cosine_similarity": score,
                "judgment": judgment or "판정 문구 없음",
                "saved_paths": {"imageA": pathA, "imageB": pathB}
            }), 200

        # c) 일반 실패
        payload = {
            "ok": False,
            "reason": "VERIFY_FAILED",
            "message": "verify_siamese 실행 실패",
            "returncode": rc,
            "saved_paths": {"imageA": pathA, "imageB": pathB}
        }
        if DEBUG_EXPOSE_ERRORS:
            payload["raw_stdout"] = stdout
            payload["raw_stderr"] = stderr
        return jsonify(payload), 500

    except Exception as e:
        traceback.print_exc()
        payload = {"ok": False, "reason": "SERVER_EXCEPTION", "message": str(e)}
        return jsonify(payload), 500


if __name__ == "__main__":
    # 인자 요구 금지: 항상 Flask 서버로 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
