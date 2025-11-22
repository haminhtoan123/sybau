# CUDA Solver Performance Optimizations

## Problem

The original CUDA solver had CPU usage spikes cycling between 100% and 0% due to inefficient resource management:

1. **Memory Allocation Overhead**: Each GPU batch allocated new buffers, copied data to GPU, ran kernel, copied back, and deallocated
2. **String Formatting Bottleneck**: `format!("{:016x}{}", nonce, suffix)` was called millions of times in the hot loop
3. **Synchronous Operations**: CPU would prepare batch (100% usage), wait for GPU (0% usage), then repeat

## Solutions Implemented

### 1. Pre-allocated GPU Buffers (`CudaAshmaizeOptimized`)

**Before**: Allocate → Copy → Compute → Copy Back → Deallocate (every batch)
```rust
// Old approach - allocates every time
let rom_buffer = device.htod_sync_copy(&rom.data)?;
let salts_buffer = device.htod_sync_copy(&all_salts)?;
// ... more allocations
```

**After**: Allocate Once → Reuse Buffers
```rust
// New approach - pre-allocated in constructor
pub struct CudaAshmaizeOptimized {
    rom_buffer: CudaSlice<u8>,           // ROM data (allocated once)
    salts_buffer: CudaSlice<u8>,         // Input nonces (reused)
    results_buffer: CudaSlice<u8>,       // Output hashes (reused)
    // ... other pre-allocated buffers
}
```

### 2. Fast Nonce Formatting

**Before**: String formatting in hot loop
```rust
let preimage = format!("{:016x}{}", nonce, suffix); // Expensive!
batch_salts.push(preimage.into_bytes());
```

**After**: Direct byte manipulation
```rust
fn write_nonce_to_bytes(&mut self, nonce: u64, salt_start: usize) {
    let hex_chars = b"0123456789abcdef";
    for i in 0..16 {
        let shift = (15 - i) * 4;
        let nibble = ((nonce >> shift) & 0xf) as usize;
        self.salt_bytes[salt_start + i] = hex_chars[nibble];
    }
}
```

### 3. Reduced Memory Transfers

**Before**: Full buffer copies every batch
```rust
device.htod_sync_copy(&all_salts)?;  // Copy entire buffer
```

**After**: Partial buffer updates
```rust
device.htod_sync_copy_into(&self.salt_bytes[..data_size], &mut self.salts_buffer)?;
```

## Usage

### Enable Optimized Solver (Default)
```bash
# Optimized solver is enabled by default
cargo run --features cuda --release -- --address 0x... --challenge_id ... --solver_mode gpu

# Explicitly enable optimizations
cargo run --features cuda --release -- --address 0x... --optimized true --solver_mode gpu
```

### Disable Optimizations (Fallback)
```bash
# Use original solver
cargo run --features cuda --release -- --address 0x... --optimized false --solver_mode gpu
```

### Solver Modes

1. **GPU Only (Optimized)**
```bash
--solver_mode gpu --optimized true
```

2. **Mixed CPU+GPU (Optimized)**
```bash
--solver_mode mixed --optimized true
```

3. **Auto-detect (Optimized)**
```bash
--solver_mode auto --optimized true  # Default
```

## Performance Improvements

### Expected Benefits

1. **Reduced CPU Usage**: Eliminates allocation/formatting spikes
2. **Higher Throughput**: Less overhead per batch = more hashes/second
3. **Better GPU Utilization**: Reduced CPU-GPU synchronization points
4. **Lower Memory Pressure**: Reused buffers instead of constant allocation

### Benchmarking

Compare performance:
```bash
# Benchmark optimized solver
cargo run --features cuda --release -- benchmark --address 0x... --optimized true

# Benchmark original solver
cargo run --features cuda --release -- benchmark --address 0x... --optimized false
```

## Technical Details

### Buffer Management

- **ROM Buffer**: Allocated once, never changes
- **Salt Buffer**: Pre-allocated to max batch size, updated with new nonces
- **Results Buffer**: Pre-allocated, reused for each batch output
- **Programs Buffer**: Pre-allocated VM program space

### Memory Layout

```
Salt Buffer Layout (per nonce):
[16 bytes hex nonce][suffix bytes][padding to max_salt_len]

Example:
0123456789abcdef<suffix_content><padding>
```

### Compatibility

- Fully backward compatible with existing solver interface
- Can be disabled via `--optimized false` flag
- Falls back to CPU if GPU initialization fails

## Troubleshooting

### If optimized solver fails:
1. Check CUDA installation and device compatibility
2. Verify sufficient GPU memory for batch size
3. Try reducing `--gpu_batch_size`
4. Use `--optimized false` to fallback to original solver

### Performance Issues:
1. Increase `--gpu_batch_size` for better GPU utilization
2. Use `--solver_mode mixed` to combine CPU+GPU
3. Monitor GPU utilization with `nvidia-smi`

## Implementation Notes

- Uses `CudaAshmaizeOptimized` struct for buffer management
- Maintains thread safety for multi-GPU setups
- Preserves exact same hash computation as original
- No changes required to CUDA kernel code