# 🚀 Instant-DB Installation Guide

Complete installation instructions for all platforms with troubleshooting tips.

## 📋 Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **pip** (included with Python)
- **libmagic** (platform-specific installation required)

## 🖥️ Platform-Specific Installation

### 🍎 macOS

#### Method 1: Homebrew (Recommended)
```bash
# Install libmagic via Homebrew
brew install libmagic

# Install Instant-DB
pip install instant-db
```

#### Method 2: MacPorts
```bash
# Install libmagic via MacPorts
sudo port install libmagic

# Install Instant-DB
pip install instant-db
```

#### Troubleshooting macOS
If you get `ImportError: failed to find libmagic`:

```bash
# Option 1: Reinstall python-magic with explicit library path
pip uninstall python-magic
LDFLAGS="-L$(brew --prefix)/lib" pip install python-magic

# Option 2: Set environment variable
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"

# Option 3: For M1/M2 Macs
arch -arm64 brew install libmagic
```

### 🐧 Linux (Ubuntu/Debian)

#### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install libmagic
sudo apt-get install libmagic1 libmagic-dev

# Install Python development headers (if needed)
sudo apt-get install python3-dev

# Install Instant-DB
pip install instant-db
```

#### CentOS/RHEL/Fedora
```bash
# For CentOS/RHEL
sudo yum install file-devel
# OR for newer versions
sudo dnf install file-devel

# Install Instant-DB
pip install instant-db
```

#### Arch Linux
```bash
# Install libmagic
sudo pacman -S file

# Install Instant-DB
pip install instant-db
```

#### Troubleshooting Linux
If you encounter permission errors:

```bash
# Install with user flag
pip install --user instant-db

# Or use virtual environment (recommended)
python3 -m venv instant_db_env
source instant_db_env/bin/activate
pip install instant-db
```

### 🪟 Windows

#### Method 1: WSL2 (Recommended)
```bash
# Install WSL2 (run in PowerShell as Administrator)
wsl --install

# Restart your computer, then in WSL2 terminal:
sudo apt-get update
sudo apt-get install libmagic1 libmagic-dev python3-pip

# Install Instant-DB
pip3 install instant-db
```

#### Method 2: Native Windows (Advanced)
```bash
# Install using conda (Anaconda/Miniconda)
conda install -c conda-forge python-magic

# Install Instant-DB
pip install instant-db
```

#### Method 3: Docker (Cross-platform)
```bash
# Pull and run Instant-DB in Docker
docker run -it --rm -v "$(pwd):/workspace" instant-db:latest

# Or build from Dockerfile
git clone https://github.com/MakeWorkTakesWork/Instant-DB
cd Instant-DB
docker build -t instant-db .
```

#### Troubleshooting Windows
For native Windows installations:

```bash
# If python-magic fails, try python-magic-bin
pip uninstall python-magic
pip install python-magic-bin

# For Conda users
conda update --all
conda install -c conda-forge libmagic
```

## 🧪 Verify Installation

After installation, verify everything works:

```bash
# Test basic import
python -c "import instant_db; print('✅ Instant-DB imported successfully')"

# Test libmagic integration
python -c "from instant_db.core.discovery import DocumentDiscovery; print('✅ Document discovery ready')"

# Test CLI
instant-db --version

# Quick functionality test
mkdir test_docs
echo "Test document content" > test_docs/sample.txt
instant-db process test_docs --auto-detect
instant-db search "test" --top-k 1
```

Expected output:
```
✅ Instant-DB imported successfully
✅ Document discovery ready
Instant-DB version 1.0.0
🔍 Auto-discovering documents in: test_docs
📊 Discovery Summary: 1 documents found
🚀 Processing 1 discovered documents...
✅ Auto-discovery processing completed!
🔍 Found 1 results for: 'test'
```

## 🔧 Virtual Environment Setup (Recommended)

Using a virtual environment prevents conflicts:

```bash
# Create virtual environment
python -m venv instant_db_env

# Activate it
# On Linux/macOS:
source instant_db_env/bin/activate
# On Windows:
instant_db_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install instant-db

# When done, deactivate
deactivate
```

## 📦 Development Installation

For contributing or development:

```bash
# Clone repository
git clone https://github.com/MakeWorkTakesWork/Instant-DB.git
cd Instant-DB

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 🐳 Docker Installation

For containerized deployment:

```bash
# Method 1: Docker Hub (when available)
docker pull instant-db:latest

# Method 2: Build from source
git clone https://github.com/MakeWorkTakesWork/Instant-DB.git
cd Instant-DB
docker build -t instant-db .

# Run with volume mounting
docker run -it --rm \
  -v "$(pwd)/documents:/workspace/documents" \
  -v "$(pwd)/output:/workspace/output" \
  instant-db process /workspace/documents
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  instant-db:
    build: .
    volumes:
      - ./documents:/workspace/documents
      - ./output:/workspace/output
      - ./config:/workspace/config
    command: ["process", "/workspace/documents", "--auto-detect"]
```

## 🚨 Common Issues & Solutions

### Issue: `ImportError: failed to find libmagic`

**Solution:**
```bash
# macOS
brew install libmagic
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"

# Linux
sudo apt-get install libmagic1 libmagic-dev

# Windows
pip install python-magic-bin
```

### Issue: `Permission denied` errors

**Solution:**
```bash
# Use user installation
pip install --user instant-db

# Or fix permissions
sudo chown -R $USER ~/.local/lib/python*/site-packages/
```

### Issue: `Command not found: instant-db`

**Solution:**
```bash
# Check if pip bin directory is in PATH
echo $PATH

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
python -m instant_db.cli --help
```

### Issue: `ModuleNotFoundError: No module named 'instant_db'`

**Solution:**
```bash
# Check Python and pip versions
python --version
pip --version

# Ensure you're using the right Python
which python
which pip

# Reinstall with explicit Python version
python3 -m pip install instant-db
```

### Issue: Large file processing fails

**Solution:**
```bash
# Increase file size limit
instant-db process docs/ --max-file-size 200

# Process in smaller batches
instant-db process docs/batch1/ --auto-detect
instant-db process docs/batch2/ --auto-detect

# Use fewer workers to reduce memory usage
instant-db process docs/ --workers 1
```

### Issue: Out of memory errors

**Solution:**
```bash
# Reduce chunk size in config
echo "chunking:
  chunk_size: 512
  chunk_overlap: 50" > .instant_db.yaml

# Process smaller batches
find docs/ -name "*.pdf" | head -10 | xargs -I {} instant-db process {}

# Use streaming for very large collections
instant-db process docs/ --batch-size 5
```

## 🔄 Updating Instant-DB

```bash
# Check current version
instant-db --version

# Update to latest version
pip install --upgrade instant-db

# Update with dependencies
pip install --upgrade --force-reinstall instant-db

# For development installations
cd Instant-DB
git pull origin main
pip install -e . --upgrade
```

## 🆘 Getting Help

If you're still having issues:

1. **Check the logs:**
   ```bash
   instant-db process docs/ --verbose
   ```

2. **Create an issue:** [GitHub Issues](https://github.com/MakeWorkTakesWork/Instant-DB/issues)

3. **Include system info:**
   ```bash
   python --version
   pip --version
   uname -a  # Linux/macOS
   systeminfo  # Windows
   ```

4. **Test with minimal example:**
   ```bash
   mkdir test && echo "test" > test/file.txt
   instant-db process test/ --verbose
   ```

## 📚 Next Steps

After successful installation:

1. **📖 Read the [README](README.md)** for basic usage
2. **🚀 Try the [Quick Start Guide](README.md#quick-start)**
3. **🔍 Explore [Advanced Features](README.md#advanced-features)**
4. **🤝 Join our [Community](https://github.com/MakeWorkTakesWork/Instant-DB/discussions)**

---

**💡 Pro Tips:**
- Always use virtual environments for Python projects
- Keep your system packages updated
- Use Docker for consistent deployments
- Check system requirements before installation 