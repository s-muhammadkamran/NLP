@echo off
REM Create Virtual Environment
python -m venv .venv

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Upgrade pip
python --trusted-host pypi.org --trusted-host files.pythonhosted.org -m pip install --upgrade pip

REM Install requirements
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

REM Download en_core_web_sm
python --trusted-host pypi.org --trusted-host files.pythonhosted.org -m spacy download en_core_web_sm

echo.
echo Environment setup complete.
pause