@echo off
REM 1) 가상환경
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate

REM 2) 의존성
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 3) 환경변수(없으면 기본값)
set "MODEL_PATH=%MODEL_PATH%"
if not defined MODEL_PATH set "MODEL_PATH=%~dp0saved_model\model_lines_Ver06.keras"
set "FH_DEBUG=1"

REM 4) 실행
python app.py