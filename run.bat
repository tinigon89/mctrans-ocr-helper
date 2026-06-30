@echo off
REM Launch the MC-Trans OCR helper. Self-contained: the CUDA/Vulkan runtime is
REM downloaded to match the hardware on first launch, so nothing CUDA needs to be
REM on PATH (the toolkit is only required to *build*).
cd /d "%~dp0"

REM In the release zip the exe sits NEXT TO this script. Fall back to the dev
REM build paths so the same script also works from a source checkout.
if exist "mctrans-ocr-helper.exe" (
  "mctrans-ocr-helper.exe"
) else if exist "target\release\mctrans-ocr-helper.exe" (
  "target\release\mctrans-ocr-helper.exe"
) else if exist "target\debug\mctrans-ocr-helper.exe" (
  "target\debug\mctrans-ocr-helper.exe"
) else (
  echo [ERROR] mctrans-ocr-helper.exe not found next to this script.
)

REM Keep the window open so first-launch download progress / crash output stays
REM visible instead of the window vanishing immediately.
echo.
echo Helper stopped. Press any key to close...
pause >nul
