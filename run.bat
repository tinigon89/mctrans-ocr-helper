@echo off
REM Launch the MC-Trans OCR helper. Self-contained: the CUDA/Vulkan runtime is
REM downloaded to match the hardware on first launch, so nothing CUDA needs to be
REM on PATH (the toolkit is only required to *build*).
cd /d "%~dp0"
target\release\mctrans-ocr-helper.exe 2>nul || target\debug\mctrans-ocr-helper.exe
