# Installation

**Setting up the Field of Truth vQbit Framework**

---

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended for large problems
- **Storage**: 10GB free space for framework + data
- **GPU**: Optional, CUDA-compatible for acceleration

### Software Requirements
- **Python**: 3.9+ (3.11 recommended)
- **Operating System**: Windows 10+, macOS 12+, Linux (Ubuntu 20.04+)
- **Git**: For repository cloning and version control

---

## Quick Installation

### Option 1: Clone from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the framework
python -c "from core.vqbit_engine import VQbitEngine; print('✅ Installation successful!')"
```

### Option 2: pip Install (Coming Soon)

```bash
# Future release will support direct pip installation
pip install fot-fluid-dynamics
```

---

## Detailed Installation Steps

### 1. Environment Setup

#### Using Conda (Recommended)
```bash
# Create conda environment
conda create -n fot-millennium python=3.11
conda activate fot-millennium

# Install scientific computing stack
conda install numpy scipy matplotlib pandas plotly
conda install -c conda-forge streamlit

# Clone and install FoT framework
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics
pip install -r requirements.txt
```

#### Using venv
```bash
# Create virtual environment
python3.11 -m venv fot-env
source fot-env/bin/activate  # Linux/macOS
# fot-env\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install framework
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics
pip install -r requirements.txt
```

### 2. Dependency Installation

#### Core Dependencies
```bash
# Essential packages (automatically installed)
pip install numpy>=2.3.3
pip install scipy>=1.16.2  
pip install matplotlib>=3.10.6
pip install pandas>=2.3.2
pip install streamlit>=1.49.1
pip install plotly>=6.3.0
```

#### Optional Dependencies
```bash
# For enhanced performance (optional)
pip install numba>=0.59.0        # JIT compilation
pip install joblib>=1.5.2        # Parallel processing

# For advanced visualization (optional)  
pip install seaborn>=0.13.2      # Statistical plots
pip install PIL>=11.3.0          # Image processing

# For development (optional)
pip install pytest>=8.4.2        # Testing framework
pip install black>=25.9.0        # Code formatting
```

### 3. GPU Acceleration (Optional)

#### NVIDIA CUDA Setup
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU acceleration
python -c "from core.vqbit_engine import VQbitEngine; engine = VQbitEngine(); print('GPU ready!' if engine.cuda_available else 'CPU only')"
```

### 4. Verification

#### Basic Functionality Test
```bash
# Run basic tests
python -m pytest tests/test_basic_functionality.py -v

# Test vQbit engine
python test_vqbit_engine.py

# Test Navier-Stokes solver
python test_navier_stokes.py

# Test Millennium solver
python test_millennium_solver.py
```

#### Interactive Test
```bash
# Launch Streamlit interface
streamlit run streamlit_app.py

# Should open browser to http://localhost:8501
# Verify all sections load correctly
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:
```bash
# Core configuration
FOT_VQBIT_DIMENSION=8096
FOT_MAX_ITERATIONS=10000
FOT_CONVERGENCE_THRESHOLD=1e-12

# Performance settings
FOT_NUM_THREADS=8
FOT_MEMORY_LIMIT=16GB
FOT_CACHE_SIZE=1000

# Paths
FOT_DATA_PATH=./data
FOT_RESULTS_PATH=./results
FOT_CACHE_PATH=./cache

# Optional: API settings
FOT_API_URL=http://localhost:8000
FOT_API_TOKEN=your_token_here
```

### Configuration File

Edit `config/default_config.json`:
```json
{
    "vqbit_engine": {
        "dimension": 8096,
        "virtue_weights": [0.25, 0.25, 0.25, 0.25],
        "coherence_threshold": 0.8,
        "max_iterations": 10000
    },
    "navier_stokes": {
        "spatial_dimension": 3,
        "time_step": 1e-4,
        "viscosity": 1e-3,
        "reynolds_number": 1000
    },
    "millennium_solver": {
        "proof_strategy": "virtue_guided",
        "verification_level": "rigorous",
        "output_format": "clay_institute"
    }
}
```

---

## Platform-Specific Instructions

### macOS Installation

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.11 git

# Install FoT framework
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics
python3.11 -m pip install -r requirements.txt

# Add streamlit to PATH (if needed)
echo 'export PATH="$HOME/Library/Python/3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Windows Installation

```powershell
# Install Python (download from python.org or use Microsoft Store)
# Install Git (download from git-scm.com)

# Clone repository
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python -c "print('FoT Framework installed successfully!')"
```

### Linux (Ubuntu/Debian) Installation

```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.11 python3.11-venv python3-pip git build-essential

# Install scientific computing libraries
sudo apt install libatlas-base-dev libopenblas-dev liblapack-dev

# Clone and install FoT framework
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics

# Launch with Docker Compose
docker-compose up -d

# Access Streamlit interface at http://localhost:8501
```

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.address", "0.0.0.0"]
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'core'
# Solution: Ensure you're in the correct directory and virtual environment is activated
cd FoTFluidDynamics
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### 2. Memory Errors
```bash
# Error: MemoryError during vQbit initialization
# Solution: Reduce vQbit dimension for systems with limited RAM
export FOT_VQBIT_DIMENSION=1024  # Instead of default 8096
```

#### 3. Streamlit Port Conflicts
```bash
# Error: Port 8501 already in use
# Solution: Use different port
streamlit run streamlit_app.py --server.port 8502
```

#### 4. CUDA Issues
```bash
# Error: CUDA out of memory
# Solution: Disable GPU acceleration
export CUDA_VISIBLE_DEVICES=""
```

### Verification Commands

```bash
# Check Python version
python --version  # Should be 3.9+

# Check key dependencies
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"

# Check FoT components
python -c "from core.vqbit_engine import VQbitEngine; print('vQbit Engine: OK')"
python -c "from core.navier_stokes_engine import NavierStokesEngine; print('NS Engine: OK')"
python -c "from core.millennium_solver import MillenniumSolver; print('Millennium Solver: OK')"
```

### Performance Tuning

#### For Low-Memory Systems
```python
# Edit config file for reduced memory usage
{
    "vqbit_engine": {
        "dimension": 1024,  # Reduced from 8096
        "batch_size": 32,   # Smaller batches
        "cache_size": 100   # Reduced cache
    }
}
```

#### For High-Performance Systems
```python
# Edit config for maximum performance
{
    "vqbit_engine": {
        "dimension": 8096,
        "num_threads": 16,    # Use all cores
        "use_gpu": true,      # Enable GPU acceleration
        "precision": "float32" # Faster computation
    }
}
```

---

## Development Setup

### For Contributors

```bash
# Clone with development dependencies
git clone https://github.com/FortressAI/FoTFluidDynamics.git
cd FoTFluidDynamics

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run full test suite
python -m pytest tests/ -v --cov=core/

# Run linting
black . && isort . && flake8
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm Configuration
- Set interpreter to `./venv/bin/python`
- Enable pytest as test runner
- Configure black as code formatter
- Set project root to FoTFluidDynamics directory

---

## Next Steps

After successful installation:

1. **Explore Examples**: Run the demo notebooks in `examples/`
2. **Read Documentation**: Check out the [[Quick Start Guide]]
3. **Try Streamlit Interface**: Launch and explore the web application
4. **Run Tests**: Verify everything works with the test suite
5. **Solve Millennium Problem**: Try solving your first instance!

---

## Getting Help

### Documentation
- **Wiki**: [GitHub Wiki](https://github.com/FortressAI/FoTFluidDynamics/wiki)
- **API Reference**: [[API Reference]]
- **Quick Start**: [[Quick Start Guide]]

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Email**: bliztafree@gmail.com for direct support
- **Discussions**: GitHub Discussions for questions

### Community
- **Contributing**: See [[Contributing Guide]]
- **Code of Conduct**: Follow our community guidelines
- **Roadmap**: Check upcoming features and improvements

---

**Author**: Rick Gillespie  
**Institution**: FortressAI Research Institute  
**Email**: bliztafree@gmail.com  
**Last Updated**: September 2025
