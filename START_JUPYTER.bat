@echo off
REM Opens the project notebook in Jupyter Lab using the "ret" conda environment
cd /d "%~dp0"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (call "%USERPROFILE%\miniconda3\Scripts\activate.bat" ret) else (call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ret)
jupyter lab RET_Digital_Twin.ipynb
pause
