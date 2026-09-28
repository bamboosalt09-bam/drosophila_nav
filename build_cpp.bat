@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
"%~dp0.venv\Scripts\python.exe" "%~dp0src\sensors\_fisheye_cpp\build.py" build_ext --inplace
