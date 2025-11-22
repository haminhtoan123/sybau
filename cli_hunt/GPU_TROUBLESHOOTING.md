# GPU Troubleshooting Guide - Solver Only Using CPU

## 🚨 Problem: GPU Solver Falls Back to CPU Only

When you run the solver with `--solver-mode gpu` but it only consumes CPU, this indicates the GPU is not being detected or initialized properly.

## 🔍 Quick Diagnosis

### Step 1: Check GPU Detection
```bash
cd rust_solver
./target/release/ashmaize-solver --solver-mode gpu --optimized true \
  --address 0x1234567890123456789012345678901234567890 \
  --challenge-id test --difficulty 00ffffff \
  --no-pre-mine test --latest-submission test --no-pre-mine-hour test
```

Look for these messages:
- ✅ "Detected X GPU(s)" - GPU found
- ❌ "No GPUs detected, falling back to CPU" - GPU not found
- ❌ "Failed to initialize CUDA" - CUDA setup issue

### Step 2: Verify CUDA Installation
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check GPU compatibility
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Step 3: Test Python Orchestrator Flags
```bash
# Try with explicit optimization flag
uv run main.py run --solver-mode gpu --optimized

# Try auto-detect mode
uv run main.py run --solver-mode auto --optimized

# Check what's actually being called
uv run main.py run --solver-mode gpu --optimized --verbose
```

## 🔧 Common Issues and Solutions

### Issue 1: CUDA Not Found During Build
**Symptoms**: 
- Build succeeds but no GPU support
- Always falls back to CPU
- No "Detected GPUs" messages

**Solution**:
```bash
# Check CUDA installation
ls -la /usr/local/cuda/
ls -la /opt/cuda/

# Set CUDA environment variables
export CUDA_PATH=/usr/local/cuda
export PATH=$PATH:$CUDA_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64

# Rebuild with CUDA
cargo clean
cargo build --features cuda --release
```

### Issue 2: GPU Not Compatible
**Symptoms**:
- CUDA installed but GPU not detected
- Older GPU model

**Solution**:
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# If compute capability < 3.5, GPU may not be supported
# Try reducing CUDA architecture in build.rs:
# Change: -arch=sm_89
# To:     -arch=sm_75  (for RTX 20xx/30xx)
# Or:     -arch=sm_61  (for GTX 10xx)
```

### Issue 3: Missing CUDA Libraries
**Symptoms**:
- Build fails with CUDA linking errors
- Runtime CUDA initialization errors

**Solution**:
```bash
# Install CUDA development libraries (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-cuda-toolkit cuda-toolkit-11-8

# Install CUDA development libraries (CentOS/RHEL)
sudo dnf install cuda-toolkit

# Verify installation
ldconfig -p | grep cuda
```

### Issue 4: Insufficient GPU Memory
**Symptoms**:
- GPU detected but fails during execution
- "CUDA out of memory" errors

**Solution**:
```bash
# Check GPU memory usage
nvidia-smi

# Use smaller batch size
uv run main.py run --solver-mode gpu --optimized --gpu-batch-size 256

# Or try auto mode
uv run main.py run --solver-mode auto --optimized
```

### Issue 5: Python Orchestrator Not Passing GPU Flags
**Symptoms**:
- Direct Rust solver works with GPU
- Python orchestrator only uses CPU

**Solution**:
```bash
# Check if orchestrator is calling the right binary
cd python_orchestrator
python -c "
import subprocess
import os
RUST_SOLVER_PATH = '../rust_solver/target/release/ashmaize-solver'
print('Binary exists:', os.path.exists(RUST_SOLVER_PATH))
result = subprocess.run([RUST_SOLVER_PATH, '--help'], capture_output=True, text=True)
print('GPU flags available:', '--solver-mode' in result.stdout and 'gpu' in result.stdout)
"

# Verify orchestrator command construction
# Edit python_orchestrator/main.py temporarily to print the command:
# Add: print('Command:', ' '.join(command))
# Before: process = subprocess.Popen(command, ...)
```

## 🎯 Step-by-Step GPU Activation

### 1. Verify Hardware
```bash
# Check NVIDIA GPU is present
lspci | grep -i nvidia

# Check GPU status
nvidia-smi
```

### 2. Install/Update CUDA
```bash
# Download CUDA installer from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

# Install CUDA
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Rebuild with CUDA Support
```bash
cd rust_solver

# Clean previous build
cargo clean
rm -rf target/

# Rebuild with CUDA
cargo build --features cuda --release

# Verify GPU support
./target/release/ashmaize-solver --help | grep -A5 -B5 "solver-mode"
```

### 4. Test GPU Functionality
```bash
# Test direct solver
./target/release/ashmaize-solver \
  --solver-mode gpu --optimized true \
  --address 0x1111111111111111111111111111111111111111 \
  --challenge-id test --difficulty 00ffffff \
  --no-pre-mine test --latest-submission test \
  --no-pre-mine-hour test --gpu-batch-size 256

# Should see: "Detected X GPU(s)" message
```

### 5. Test Python Orchestrator
```bash
cd ../python_orchestrator

# Test with explicit flags
uv run main.py run --solver-mode gpu --optimized --gpu-batch-size 256
```

## 🔍 Advanced Debugging

### Debug CUDA Detection
```bash
# Check CUDA runtime version
nvcc --version
cat /usr/local/cuda/version.txt

# Check cudarc library compatibility
cd rust_solver
cargo tree | grep cudarc

# Test basic CUDA functionality
python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv'], 
                       capture_output=True, text=True)
print('GPU Info:')
print(result.stdout)
"
```

### Check Rust Solver Log Output
```bash
# Run with verbose logging
RUST_LOG=debug ./target/release/ashmaize-solver \
  --solver-mode gpu --optimized true \
  --address 0x1111111111111111111111111111111111111111 \
  --challenge-id test --difficulty 00ffffff \
  --no-pre-mine test --latest-submission test \
  --no-pre-mine-hour test
```

### Verify Python Orchestrator Command
Temporarily modify `python_orchestrator/main.py` in the `_solve_one_challenge` function:
```python
# Add this line before: process = subprocess.Popen(command, ...)
print("🔍 DEBUG - Command being executed:")
print(" ".join(command))
print("🔍 DEBUG - Working directory:", os.getcwd())
print("🔍 DEBUG - Binary exists:", os.path.exists(command[0]))
```

## 🚀 Quick Fixes to Try

### 1. Force GPU Mode
```bash
# Try mixed mode instead of pure GPU
uv run main.py run --solver-mode mixed --optimized

# Try auto-detect
uv run main.py run --solver-mode auto --optimized
```

### 2. Reduce Memory Requirements
```bash
# Use minimal batch size
uv run main.py run --solver-mode gpu --optimized --gpu-batch-size 64
```

### 3. Update GPU Architecture
Edit `rust_solver/build.rs` and change:
```rust
// From:
"-arch=sm_89",
// To (for older GPUs):
"-arch=sm_75",  // RTX 20xx/30xx series
// Or:
"-arch=sm_61",  // GTX 10xx series
```

### 4. Check Python Environment
```bash
# Ensure using correct Python/UV environment
which python
which uv
uv --version

# Check if in correct directory
pwd  # Should be in python_orchestrator/
ls -la  # Should see main.py
```

## ✅ Success Indicators

You'll know GPU is working when you see:
- ✅ "Detected X GPU(s)" in solver output
- ✅ High GPU utilization in `nvidia-smi`
- ✅ Solver output mentions GPU device info
- ✅ Much higher hashrate than CPU-only mode
- ✅ CPU usage stays reasonable (not 100%)

## ❌ Failure Indicators

GPU is NOT working if you see:
- ❌ "No GPUs detected, falling back to CPU"
- ❌ "Failed to initialize CUDA"
- ❌ Only CPU cores at 100%, GPU at 0%
- ❌ No GPU device info in logs
- ❌ Same performance as `--solver-mode cpu`

## 🆘 If All Else Fails

1. **Use CPU mode temporarily**:
   ```bash
   uv run main.py run --solver-mode cpu
   ```

2. **Test on different system** with known-good CUDA setup

3. **Check GPU compatibility** at https://developer.nvidia.com/cuda-gpus

4. **File an issue** with your system details:
   - GPU model: `nvidia-smi --query-gpu=name --format=csv,noheader`
   - CUDA version: `nvcc --version`
   - Driver version: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
   - OS: `uname -a`
   - Build output: `cargo build --features cuda --release 2>&1`

The most common cause is CUDA not being properly installed or the GPU not being compatible with the CUDA architecture specified in the build configuration.