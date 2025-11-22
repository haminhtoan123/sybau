# Python Orchestrator

Multi-wallet mining orchestrator for the Midnight Scavenger Hunt with a Terminal UI (TUI) extended wit CUDA GPU support.

## WARNING
THIS CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

## Overview

The Python orchestrator manages mining operations across multiple wallets, automatically fetching challenges, solving them using the Rust solver, and submitting solutions. It provides a real-time terminal interface showing challenge status, mining statistics, and logs.

## Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager
- Built Rust solver at `../rust_solver/target/release/ashmaize-solver`

### Bulding the Rust solver

Add the `--features cuda` flag to the `cargo build` command to build the solver with CUDA support.

```bash
cd ../rust_solver
cargo build --release --features cuda
```


### Initial Setup

1. Initialize the database with address JSON files:

```bash
uv run main.py init addresses.json
```

Example `addresses.json` format:
```json
{
    "addresses": [
        "addr1qx2kd28nq8ac7qs93f6x2xfar6xmhz8eqm8v4s4vy2u9sek8sfxy9g2hjcwdr8cxzr7yvu6c8qy0z2yz4m3r6x8v7qzq8h8c",
        "addr1q9f6r2x8v4s4vy2u9sek8sfxy9g2hjcwdr8cxzr7yvu6c8qy0z2yz4m3r6x8v7qzq8h8ckd28nq8ac7qs93f6x2xfar6xmhz"
    ]
}
```

2. Start the orchestrator:

```bash
uv run main.py run
```

## Usage

### Initialize Database

```bash
uv run main.py init <json_files...>
```

Imports wallet addresses from JSON files with simplified format. Creates or updates `challenges.json`.

### Run Orchestrator

```bash
uv run main.py run [options]
```

#### Core Options

- `--max-solvers <N>` - Number of concurrent solvers (default: 2)
- `--challenge-selection {first,last}` - Challenge priority (default: first)
- `--solver-mode {cpu,gpu,auto,mixed}` - Solver execution mode (default: auto)

#### Timing Options

- `--solve-interval <seconds>` - Check for new challenges every N seconds (default: 120)
- `--save-interval <seconds>` - Save database to disk every N seconds (default: 600)
- `--stats-interval <seconds>` - Update wallet statistics every N seconds (default: 3600)

### Solver Modes

The `--solver-mode` flag controls how mining is performed:

- **`cpu`** - CPU-only mining (5 threads per solver)
- **`gpu`** - GPU-only mining (auto-detects and uses all available GPUs)
- **`auto`** (default) - Automatically use GPUs if available, fallback to CPU
- **`mixed`** - CPU threads + all GPUs working together simultaneously

**Implementation Status**: ✅ **COMPLETE** - All solver modes are fully implemented in both Python orchestrator and Rust solver. Multi-GPU auto-detection works seamlessly without user configuration.

#### Examples

```bash
# Default: Auto-detect and use GPUs
uv run main.py run

# CPU-only mining (useful for testing or mixed workloads)
uv run main.py run --solver-mode cpu

# GPU-only with 4 parallel solvers (max performance on multi-GPU systems)
uv run main.py run --solver-mode gpu --max-solvers 4

# Mixed mode: CPU + GPU together in single solver (maximize solutions)
uv run main.py run --solver-mode mixed --max-solvers 1

# Custom intervals: check for challenges every 30s, save every 5 minutes
uv run main.py run --solve-interval 30 --save-interval 300
```

## Architecture

### Components

1. **Fetcher Worker** - Polls API every 10 minutes for new challenges
2. **Solver Worker** - Manages parallel solver processes using ThreadPoolExecutor
3. **Saver Worker** - Periodically writes database to disk
4. **Stats Worker** - Updates wallet mining statistics from API

### Data Files

- `challenges.json` - Main database (addresses, challenges, statistics)
- `challenges.json.journal` - Write-ahead log for crash recovery
- `orchestrator.log` - Detailed logging output

### TUI Interface

- **Top Panel** - Challenge table showing status for each wallet (✅ validated, ⚙️ solving, ⏳ available, ❌ expired)
- **Bottom Left** - Live log viewer with timestamped events
- **Bottom Right** - Wallet statistics (receipts, NIGHT tokens)
- **Footer** - Keyboard shortcuts (Ctrl+C to quit)

## Multi-GPU Performance

The orchestrator works seamlessly with multi-GPU systems:

- **GPU Mode**: Each solver process auto-detects and uses all GPUs (batch size: 131,072 per GPU)
- **Optimal Setup**: 4x RTX 4090 achieves ~286k H/s (71k H/s per GPU)
- **Auto-Detection**: Rust solver automatically detects and uses all available GPUs without configuration
- **Compatibility**: Supports RTX 3000/4000/5000 series (sm_86/89/90)
- **Hashrate Calculation**: For GPU modes, hashrate is estimated based on batch rounds completed (more accurate than nonce-based calculation)

## Development

### Running Tests

```bash
cd python_orchestrator
uv run pytest tests/
```

### Code Coverage

```bash
uv run coverage run -m pytest tests/
uv run coverage report
```

## Troubleshooting

### "Database file not found"

Run `uv run main.py init <json_files>` first to create the database.

### "Rust solver error"

Ensure the Rust solver is built:

```bash
cd ../rust_solver
cargo build --release --features cuda
```

### GPU Not Detected

- Verify CUDA installation: `nvidia-smi`
- Check solver output for GPU detection messages
- Try `--solver-mode gpu` to force GPU mode (will error if no GPUs available)
  - There are other solver modes, but other than pure CPU or pure GPU, they are not tested yet.

