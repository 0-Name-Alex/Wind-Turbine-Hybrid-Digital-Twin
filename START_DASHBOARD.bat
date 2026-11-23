@echo off
REM Starts the digital twin dashboard in your browser (http://localhost:8501)
cd /d "%~dp0"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (call "%USERPROFILE%\miniconda3\Scripts\activate.bat" ret) else (call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ret)
streamlit run digital_twin_app.py
pause
