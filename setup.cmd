@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo [1/6] Detect Python...
set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo ERROR: Python was not found. Please install Python 3.10+ first.
    exit /b 1
)

echo Using Python launcher: %PY_CMD%

echo [2/6] Create virtual environment .venv...
if not exist ".venv" (
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create .venv
        exit /b 1
    )
) else (
    echo .venv already exists, reusing it.
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate .venv
    exit /b 1
)

echo [3/6] Upgrade pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip tools
    exit /b 1
)

echo [4/6] Install PyTorch CUDA 12.1 build...
python -m pip install --upgrade torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install torch CUDA build
    exit /b 1
)

echo [5/6] Install DGL CUDA + project packages...
python -m pip install --upgrade dgl==2.2.1 -f https://data.dgl.ai/wheels/cu121/repo.html
if errorlevel 1 (
    echo ERROR: Failed to install dgl CUDA build
    exit /b 1
)

python -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
if errorlevel 1 (
    echo ERROR: Failed to install torch_scatter
    exit /b 1
)

python -m pip install --upgrade torchdata==0.7.1
if errorlevel 1 (
    echo ERROR: Failed to install compatible torchdata version
    exit /b 1
)

python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages from requirements.txt
    exit /b 1
)

echo [6/6] Verify environment...
python -c "import torch, dgl, dgllife, rdkit, numpy, pandas, sklearn, tqdm, scipy; print('Python OK'); print('torch:', torch.__version__); print('torch.cuda.is_available:', torch.cuda.is_available()); print('torch.cuda.version:', torch.version.cuda); print('dgl:', dgl.__version__); print('dgllife:', dgllife.__version__); print('rdkit:', rdkit.__version__); print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('sklearn:', sklearn.__version__); print('tqdm:', tqdm.__version__); print('scipy:', scipy.__version__)"
if errorlevel 1 (
    echo ERROR: Verify failed. Please check package installation logs above.
    exit /b 1
)

echo.
echo Setup complete.
echo Continue to activate environment with: .venv\Scripts\activate

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

exit /b 0
