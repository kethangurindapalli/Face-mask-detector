@echo off
REM Use Python 3.13 (TensorFlow is not compatible with 3.14)
set PYTHON=C:\Users\kethan.G\AppData\Local\Programs\Python\Python313\python.exe
if "%1"=="" (
    echo Usage: run.bat [command]
    echo Commands:
    echo   setup    - Download dataset and face detector
    echo   train    - Train the model
    echo   detect   - Run real-time webcam detection
    echo   app      - Start Flask web server
    echo.
    echo Examples:
    echo   run.bat app
    echo   run.bat detect
    goto :eof
)
if /i "%1"=="setup" "%PYTHON%" setup_data.py
if /i "%1"=="train" "%PYTHON%" train.py
if /i "%1"=="detect" "%PYTHON%" detect.py
if /i "%1"=="app" "%PYTHON%" app.py
