@echo off
REM Launch the MC-Trans OCR helper (CUDA build).
REM candle's cudarc was built against the CUDA toolkit (cublas64_12.dll etc.), so
REM the toolkit's bin must be on PATH. llama.cpp brings its own CUDA 13 DLLs.
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "%~dp0"
target\release\mctrans-ocr-helper.exe 2>nul || target\debug\mctrans-ocr-helper.exe
