# Project Setup

## Install uv

For `uv` installation instructions, please visit the official documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

## Environment Configuration

### 1. Create Virtual Environment

```bash
uv venv --python 3.11
```

### 2. Activate Virtual Environment

-   **Windows (PowerShell):**
    ```powershell
    .venv\\Scripts\\Activate.ps1
    ```
-   **Windows (CMD):**
    ```cmd
    .venv\\Scripts\\activate.bat
    ```
-   **Linux / macOS (bash/zsh):**
    ```bash
    source .venv/bin/activate
    ```

### 3. Install Dependencies

Execute the following commands in order:

```bash
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
uv pip install tqdm==4.67.1 --index-strategy unsafe-best-match
```
```bash
uv pip install matplotlib==3.10.0 --index-strategy unsafe-best-match
```
```bash
uv pip install imageio==2.37.0 --index-strategy unsafe-best-match
```
```bash
uv pip install numpy==1.26.4 --index-strategy unsafe-best-match
```
