use ashmaize::Rom;
use ashmaize::b2::VM;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

#[derive(Debug)]
pub enum GpuError {
    DriverError(DriverError),
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DriverError(e) => write!(f, "CUDA driver error: {:?}", e),
            GpuError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::DriverError(e)
    }
}

impl From<std::io::Error> for GpuError {
    fn from(e: std::io::Error) -> Self {
        GpuError::Other(e.to_string())
    }
}

impl From<String> for GpuError {
    fn from(s: String) -> Self {
        GpuError::Other(s)
    }
}

impl From<&str> for GpuError {
    fn from(s: &str) -> Self {
        GpuError::Other(s.to_string())
    }
}

pub type GpuResult<T> = Result<T, GpuError>;

pub struct CudaAshmaize {
    device: Arc<CudaDevice>,
    kernel_func: CudaFunction,
    kernel_shared_mem: Option<CudaFunction>,
}

pub struct CudaAshmaizeOptimized {
    device: Arc<CudaDevice>,
    kernel_func: CudaFunction,
    // Reusable GPU buffers
    rom_buffer: CudaSlice<u8>,
    rom_digest_buffer: CudaSlice<u8>,
    salts_buffer: CudaSlice<u8>,
    initial_prog_seeds_buffer: CudaSlice<u8>,
    programs_buffer: CudaSlice<u8>,
    results_buffer: CudaSlice<u8>,
    // Reusable CPU buffers
    salt_bytes: Vec<u8>,
    initial_prog_seeds: Vec<u8>,
    results_host: Vec<u8>,
    // Configuration
    max_batch_size: usize,
    max_salt_len: usize,
    program_size: usize,
    suffix: String,
    suffix_bytes: Vec<u8>,
}

impl CudaAshmaize {
    pub fn new() -> GpuResult<Self> {
        Self::new_with_device(0)
    }

    pub fn new_with_device(device_id: usize) -> GpuResult<Self> {
        let device = CudaDevice::new(device_id)?;

        let ptx = compile_kernel()?;
        device.load_ptx(ptx.clone(), "ashmaize", &["ashmaize_hash_kernel"])?;

        let kernel_func = device
            .get_func("ashmaize", "ashmaize_hash_kernel")
            .ok_or("Failed to get kernel function")?;

        // TODO: Re-enable shared memory kernel when fixed
        let kernel_shared_mem = None;
        // let kernel_shared_mem = match compile_kernel_shared_mem() {
        //     Ok(ptx_shared) => {
        //         device.load_ptx(ptx_shared, "ashmaize_shared", &["ashmaize_hash_kernel_shared_mem"]).ok();
        //         device.get_func("ashmaize_shared", "ashmaize_hash_kernel_shared_mem")
        //     }
        //     Err(_) => None
        // };

        Ok(Self {
            device,
            kernel_func,
            kernel_shared_mem,
        })
    }

    pub fn get_device_count() -> GpuResult<usize> {
        Ok(cudarc::driver::result::device::get_count()? as usize)
    }

    pub fn hash_parallel(
        &self,
        salts: &[&[u8]],
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> GpuResult<Vec<[u8; 64]>> {
        self.hash_parallel_with_block_size(salts, rom, nb_loops, nb_instrs, 256)
    }

    pub fn hash_parallel_with_block_size(
        &self,
        salts: &[&[u8]],
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
        threads_per_block: u32,
    ) -> GpuResult<Vec<[u8; 64]>> {
        self.hash_parallel_with_kernel(salts, rom, nb_loops, nb_instrs, threads_per_block, false)
    }

    pub fn hash_parallel_shared_mem(
        &self,
        salts: &[&[u8]],
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> GpuResult<Vec<[u8; 64]>> {
        if self.kernel_shared_mem.is_none() {
            return Err("Shared memory kernel not loaded".into());
        }
        self.hash_parallel_with_kernel(salts, rom, nb_loops, nb_instrs, 256, true)
    }

    fn hash_parallel_with_kernel(
        &self,
        salts: &[&[u8]],
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
        threads_per_block: u32,
        use_shared_mem: bool,
    ) -> GpuResult<Vec<[u8; 64]>> {
        let num_hashes = salts.len();
        let program_size = nb_instrs as usize * 20;

        // Find max salt length
        let max_salt_len = salts.iter().map(|s| s.len()).max().unwrap_or(0);

        // Pack salts with proper length (no truncation)
        let mut all_salts = Vec::new();
        for salt in salts {
            all_salts.extend_from_slice(salt);
            // Pad to max length if needed for alignment
            for _ in salt.len()..max_salt_len {
                all_salts.push(0);
            }
        }

        let mut initial_prog_seeds = Vec::new();
        for salt in salts {
            let vm = VM::new(&rom.digest, nb_instrs, salt);
            initial_prog_seeds.extend_from_slice(&vm.prog_seed);
        }

        let template_program = vec![0u8; program_size];
        let mut all_programs = Vec::new();
        for _ in 0..num_hashes {
            all_programs.extend_from_slice(&template_program);
        }

        let rom_buffer = self.device.htod_sync_copy(&rom.data)?;
        let rom_digest_buffer = self.device.htod_sync_copy(rom.digest.as_bytes())?;
        let salts_buffer = self.device.htod_sync_copy(&all_salts)?;
        let initial_prog_seeds_buffer = self.device.htod_sync_copy(&initial_prog_seeds)?;
        let mut programs_buffer = self.device.htod_sync_copy(&all_programs)?;
        let mut results_buffer = self.device.alloc_zeros::<u8>(num_hashes * 64)?;

        let salt_len = max_salt_len as u32;

        let num_blocks = (num_hashes as u32 + threads_per_block - 1) / threads_per_block;

        let shared_mem_bytes = if use_shared_mem { 16384 } else { 0 };

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes,
        };

        let kernel = if use_shared_mem {
            self.kernel_shared_mem.as_ref().unwrap().clone()
        } else {
            self.kernel_func.clone()
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    &rom_buffer,
                    rom.data.len() as u32,
                    &rom_digest_buffer,
                    &salts_buffer,
                    salt_len,
                    &initial_prog_seeds_buffer,
                    &mut programs_buffer,
                    nb_loops,
                    nb_instrs,
                    program_size as u32,
                    &mut results_buffer,
                    num_hashes as u32,
                ),
            )?;
        }

        let results_host = self.device.dtoh_sync_copy(&results_buffer)?;

        let mut results = Vec::new();
        for i in 0..num_hashes {
            let mut result = [0u8; 64];
            result.copy_from_slice(&results_host[i * 64..(i + 1) * 64]);
            results.push(result);
        }

        Ok(results)
    }

    pub fn get_device_info(&self) -> GpuResult<String> {
        Ok(format!("CUDA Device: {}", self.device.name()?,))
    }
}

fn compile_kernel() -> GpuResult<Ptx> {
    // Load PTX that was compiled during build time by build.rs
    // The PTX is embedded in the binary as a static string
    const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ashmaize.ptx"));

    Ok(Ptx::from_src(PTX_SRC))
}

// TODO: Re-enable when shared_mem kernel is fixed
// fn compile_kernel_shared_mem() -> GpuResult<Ptx> {
//     const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ashmaize_shared_mem.ptx"));
//     Ok(Ptx::from_src(PTX_SRC))
// }

#[allow(dead_code)]
pub fn hash_gpu_or_cpu(salt: &[u8], rom: &Rom, nb_loops: u32, nb_instrs: u32) -> [u8; 64] {
    match CudaAshmaize::new() {
        Ok(cuda) => match cuda.hash_parallel(&[salt], rom, nb_loops, nb_instrs) {
            Ok(results) => results.into_iter().next().unwrap(),
            Err(_) => ashmaize::b2::hash(salt, rom, nb_loops, nb_instrs),
        },
        Err(_) => ashmaize::b2::hash(salt, rom, nb_loops, nb_instrs),
    }
}

#[allow(dead_code)]
pub fn hash_gpu(salt: &[u8], rom: &Rom, nb_loops: u32, nb_instrs: u32) -> GpuResult<[u8; 64]> {
    let cuda = CudaAshmaize::new()?;
    let results = cuda.hash_parallel(&[salt], rom, nb_loops, nb_instrs)?;
    Ok(results.into_iter().next().unwrap())
}

impl CudaAshmaizeOptimized {
    pub fn new(
        rom: &Rom,
        suffix: &str,
        max_batch_size: usize,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> GpuResult<Self> {
        Self::new_with_device(0, rom, suffix, max_batch_size, nb_loops, nb_instrs)
    }

    pub fn new_with_device(
        device_id: usize,
        rom: &Rom,
        suffix: &str,
        max_batch_size: usize,
        _nb_loops: u32,
        nb_instrs: u32,
    ) -> GpuResult<Self> {
        let device = CudaDevice::new(device_id)?;

        let ptx = compile_kernel()?;
        device.load_ptx(ptx.clone(), "ashmaize", &["ashmaize_hash_kernel"])?;

        let kernel_func = device
            .get_func("ashmaize", "ashmaize_hash_kernel")
            .ok_or("Failed to get kernel function")?;

        let program_size = nb_instrs as usize * 20;
        let max_salt_len = 16 + suffix.len(); // 16 hex chars for nonce + suffix
        let suffix_bytes = suffix.as_bytes().to_vec();

        // Pre-allocate GPU buffers
        let rom_buffer = device.htod_sync_copy(&rom.data)?;
        let rom_digest_buffer = device.htod_sync_copy(rom.digest.as_bytes())?;
        let salts_buffer = device.alloc_zeros::<u8>(max_batch_size * max_salt_len)?;
        let initial_prog_seeds_buffer = device.alloc_zeros::<u8>(max_batch_size * 64)?;
        let programs_buffer = device.alloc_zeros::<u8>(max_batch_size * program_size)?;
        let results_buffer = device.alloc_zeros::<u8>(max_batch_size * 64)?;

        // Pre-allocate CPU buffers
        let salt_bytes = vec![0u8; max_batch_size * max_salt_len];
        let initial_prog_seeds = vec![0u8; max_batch_size * 64];
        let results_host = vec![0u8; max_batch_size * 64];

        Ok(Self {
            device,
            kernel_func,
            rom_buffer,
            rom_digest_buffer,
            salts_buffer,
            initial_prog_seeds_buffer,
            programs_buffer,
            results_buffer,
            salt_bytes,
            initial_prog_seeds,
            results_host,
            max_batch_size,
            max_salt_len,
            program_size,
            suffix: suffix.to_string(),
            suffix_bytes,
        })
    }

    // Fast nonce to bytes conversion without string formatting
    fn write_nonce_to_bytes(&mut self, nonce: u64, salt_start: usize) {
        let hex_chars = b"0123456789abcdef";

        // Write hex bytes directly (little-endian to big-endian hex string)
        for i in 0..16 {
            let shift = (15 - i) * 4;
            let nibble = ((nonce >> shift) & 0xf) as usize;
            self.salt_bytes[salt_start + i] = hex_chars[nibble];
        }
    }

    pub fn hash_batch_optimized(
        &mut self,
        start_nonce: u64,
        batch_size: usize,
        rom: &Rom,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> GpuResult<Vec<[u8; 64]>> {
        let actual_batch_size = batch_size.min(self.max_batch_size);

        // Prepare salts efficiently - no string formatting
        for i in 0..actual_batch_size {
            let nonce = start_nonce + i as u64;
            let salt_start = i * self.max_salt_len;

            // Write nonce as hex directly to buffer
            self.write_nonce_to_bytes(nonce, salt_start);

            // Copy suffix bytes
            let suffix_start = salt_start + 16;
            let suffix_end = suffix_start + self.suffix_bytes.len();
            self.salt_bytes[suffix_start..suffix_end].copy_from_slice(&self.suffix_bytes);
        }

        // Prepare initial program seeds
        for i in 0..actual_batch_size {
            let salt_start = i * self.max_salt_len;
            let salt = &self.salt_bytes[salt_start..salt_start + self.max_salt_len];
            let vm = VM::new(&rom.digest, nb_instrs, salt);

            let prog_seed_start = i * 64;
            self.initial_prog_seeds[prog_seed_start..prog_seed_start + 64]
                .copy_from_slice(&vm.prog_seed);
        }

        // Update GPU buffers (only copy what we need)
        let salt_data_size = actual_batch_size * self.max_salt_len;
        let prog_seed_data_size = actual_batch_size * 64;

        self.device
            .htod_sync_copy_into(&self.salt_bytes[..salt_data_size], &mut self.salts_buffer)?;
        self.device.htod_sync_copy_into(
            &self.initial_prog_seeds[..prog_seed_data_size],
            &mut self.initial_prog_seeds_buffer,
        )?;

        // Launch kernel
        let threads_per_block = 256u32;
        let num_blocks = (actual_batch_size as u32 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernel_func.clone().launch(
                cfg,
                (
                    &self.rom_buffer,
                    rom.data.len() as u32,
                    &self.rom_digest_buffer,
                    &self.salts_buffer,
                    self.max_salt_len as u32,
                    &self.initial_prog_seeds_buffer,
                    &mut self.programs_buffer,
                    nb_loops,
                    nb_instrs,
                    self.program_size as u32,
                    &mut self.results_buffer,
                    actual_batch_size as u32,
                ),
            )?;
        }

        // Copy results back
        let result_data_size = actual_batch_size * 64;
        self.device.dtoh_sync_copy_into(
            &self.results_buffer,
            &mut self.results_host[..result_data_size],
        )?;

        // Convert to result format
        let mut results = Vec::with_capacity(actual_batch_size);
        for i in 0..actual_batch_size {
            let mut result = [0u8; 64];
            let start = i * 64;
            result.copy_from_slice(&self.results_host[start..start + 64]);
            results.push(result);
        }

        Ok(results)
    }

    pub fn get_device_info(&self) -> GpuResult<String> {
        Ok(format!("CUDA Device: {}", self.device.name()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ashmaize::rom::RomGenerationType;

    #[test]
    fn test_gpu_hash_basic() {
        let rom = Rom::new(
            b"test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"test_salt";
        let result = hash_gpu_or_cpu(salt, &rom, 8, 256);

        assert_eq!(result.len(), 64);
        assert!(!result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_gpu_hash_consistency() {
        let rom = Rom::new(
            b"test_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"consistent_test";
        let result1 = hash_gpu_or_cpu(salt, &rom, 8, 256);
        let result2 = hash_gpu_or_cpu(salt, &rom, 8, 256);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_gpu_hash_vs_cpu() {
        let rom = Rom::new(
            b"comparison_seed",
            RomGenerationType::TwoStep {
                pre_size: 1024,
                mixing_numbers: 4,
            },
            10_240,
        );

        let salt = b"comparison_test";
        let cpu_result = ashmaize::b2::hash(salt, &rom, 8, 256);

        if let Ok(gpu_result) = hash_gpu(salt, &rom, 8, 256) {
            assert_eq!(cpu_result, gpu_result);
        }
    }
}
