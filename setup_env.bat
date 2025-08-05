@echo off
REM Create Virtual Environment
if exist ".venv" (
    echo Virtual Environment already exists.
) else (
    python -m venv .venv
    echo Virtual Environment created in .venv
)

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Install requirements
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

REM Upgrade pip
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

REM Download en_core_web_sm
python -m spacy download en_core_web_sm --trusted-host pypi.org --trusted-host files.pythonhosted.org

echo.
echo Environment setup complete.
pause