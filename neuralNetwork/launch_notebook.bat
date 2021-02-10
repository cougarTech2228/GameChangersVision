@echo off
if exist venv goto launch
python -m pip install virtualenv
echo creating virtual env
python -m venv venv
IF ERRORLEVEL 1 goto :error
echo Created
echo Installing Reqiured Packages...
venv\Scripts\pip.exe install -r requirements.txt

:launch
venv\Scripts\pip.exe install -r requirements.txt -q
IF ERRORLEVEL 1 goto :error
venv\Scripts\jupyter notebook GalacticSearch-MobileNetV2.ipynb
pause
goto :EOF

:error
echo FAILED
pause
