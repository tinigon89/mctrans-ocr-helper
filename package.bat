@echo off
setlocal
REM ---------------------------------------------------------------------------
REM Package a release of the MC-Trans OCR helper.
REM Output: dist\mctrans-ocr-helper-windows-cuda.zip  (~80 MB, self-contained).
REM No CUDA DLLs are bundled — the runtime is downloaded per-hardware on first
REM launch (CUDA 13.1 / Vulkan), exactly like koharu.
REM
REM Adjust the toolchain paths below to your machine if they differ.
REM ---------------------------------------------------------------------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=H:\NVIDIA\CUDA13.1"
set "PATH=%CUDA_PATH%\bin;C:\Program Files\CMake\bin;C:\Program Files\LLVM\bin;%PATH%"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
cd /d "%~dp0"

echo === nvcc ===
nvcc --version | findstr release

echo === building release (--features cuda) ===
cargo build --release --features cuda
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)

echo === assembling dist ===
set "DIST=dist\mctrans-ocr-helper-windows-cuda"
if exist "%DIST%" rmdir /s /q "%DIST%"
mkdir "%DIST%"
copy /y "target\release\mctrans-ocr-helper.exe" "%DIST%\" >nul
copy /y "run.bat" "%DIST%\" >nul
copy /y "README.md" "%DIST%\" >nul

echo === zipping ===
powershell -NoProfile -Command "Compress-Archive -Force -Path '%DIST%\*' -DestinationPath 'dist\mctrans-ocr-helper-windows-cuda.zip'"

echo.
echo Done: dist\mctrans-ocr-helper-windows-cuda.zip
echo Upload it to the helper repo's GitHub Releases.
