// use ashmaize::{hash, Rom, RomGenerationType};
use ashmaize::b2::hash; // Use the blake2 implementation (slightly faster)
use ashmaize::{Rom, RomGenerationType};
use clap::Parser;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const NUM_THREADS: u64 = 5;
pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;
const OPTIMAL_BATCH_PER_GPU: usize = 131072;

mod tests;

#[cfg(feature = "cuda")]
mod gpu;

#[cfg(feature = "cuda")]
mod benchmark;

#[derive(clap::ValueEnum, Clone, Debug)]
enum SolverMode {
    Cpu,
    Gpu,
    Auto,
    Mixed,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long)]
    address: Option<String>,
    #[arg(long)]
    challenge_id: Option<String>,
    #[arg(long)]
    difficulty: Option<String>,
    #[arg(long)]
    no_pre_mine: Option<String>,
    #[arg(long)]
    latest_submission: Option<String>,
    #[arg(long)]
    no_pre_mine_hour: Option<String>,

    #[cfg(feature = "cuda")]
    #[arg(long, value_enum, default_value = "auto")]
    solver_mode: SolverMode,

    #[cfg(feature = "cuda")]
    #[arg(long, default_value = "false")]
    use_gpu: bool,

    #[cfg(feature = "cuda")]
    #[arg(long, default_value = "0")]
    gpu_id: usize,

    #[cfg(feature = "cuda")]
    #[arg(long, default_value_t = OPTIMAL_BATCH_PER_GPU)]
    gpu_batch_size: usize,

    #[cfg(feature = "cuda")]
    #[arg(
        long,
        default_value = "true",
        help = "Use optimized GPU solver with pre-allocated buffers"
    )]
    optimized: bool,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    #[cfg(feature = "cuda")]
    Benchmark {
        #[arg(long)]
        address: String,
        #[arg(long)]
        challenge_id: String,
        #[arg(long)]
        difficulty: String,
        #[arg(long)]
        no_pre_mine: String,
        #[arg(long)]
        latest_submission: String,
        #[arg(long)]
        no_pre_mine_hour: String,
        #[arg(long, default_value = "10")]
        duration_secs: u64,
        #[arg(long, default_value = "1024")]
        batch_size: usize,
        #[arg(long, default_value = "false")]
        compare: bool,
    },
    #[cfg(feature = "cuda")]
    TestHash {
        #[arg(long)]
        address: String,
        #[arg(long)]
        challenge_id: String,
        #[arg(long)]
        difficulty: String,
        #[arg(long)]
        no_pre_mine: String,
        #[arg(long)]
        latest_submission: String,
        #[arg(long)]
        no_pre_mine_hour: String,
        #[arg(long)]
        nonce: String,
    },
}

pub fn hash_structure_good(hash: &[u8], difficulty_mask: u32) -> bool {
    if hash.len() < 4 {
        return false; // Not enough bytes to apply a u32 mask
    }

    let hash_prefix = u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]);
    (hash_prefix & !difficulty_mask) == 0
}

pub fn init_rom(no_pre_mine_hex: &str) -> Rom {
    Rom::new(
        no_pre_mine_hex.as_bytes(),
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        1 * GB,
    )
}

fn main() {
    let args = Args::parse();

    match args.command {
        #[cfg(feature = "cuda")]
        Some(Command::Benchmark {
            address,
            challenge_id,
            difficulty,
            no_pre_mine,
            latest_submission,
            no_pre_mine_hour,
            duration_secs,
            batch_size,
            compare,
        }) => {
            let config = benchmark::BenchmarkConfig {
                address,
                challenge_id,
                difficulty,
                no_pre_mine,
                latest_submission,
                no_pre_mine_hour,
                duration_secs,
                batch_size,
            };

            if compare {
                benchmark::benchmark_cpu_vs_gpu(&config).unwrap();
            } else {
                let result = benchmark::benchmark_gpu_hashrate(&config).unwrap();
                result.print_summary("GPU");
            }
        }
        #[cfg(feature = "cuda")]
        Some(Command::TestHash {
            address,
            challenge_id,
            difficulty,
            no_pre_mine,
            latest_submission,
            no_pre_mine_hour,
            nonce,
        }) => {
            let nonce_val = u64::from_str_radix(&nonce, 16).unwrap();
            if let Err(e) = benchmark::run_single_hash_test(
                &address,
                &challenge_id,
                &difficulty,
                &no_pre_mine,
                &latest_submission,
                &no_pre_mine_hour,
                nonce_val,
            ) {
                eprintln!("Error during hash test: {:?}", e);
                std::process::exit(1);
            }
        }
        None => {
            let address = args.address.expect("--address required");
            let challenge_id = args.challenge_id.expect("--challenge_id required");
            let difficulty = args.difficulty.expect("--difficulty required");
            let no_pre_mine = args.no_pre_mine.expect("--no_pre_mine required");
            let latest_submission = args
                .latest_submission
                .expect("--latest_submission required");
            let no_pre_mine_hour = args.no_pre_mine_hour.expect("--no_pre_mine_hour required");

            #[cfg(feature = "cuda")]
            {
                let mut mode = args.solver_mode;
                if args.use_gpu {
                    mode = SolverMode::Gpu;
                }
                let gpu_batch_size = args.gpu_batch_size;

                solve(
                    &address,
                    &challenge_id,
                    &difficulty,
                    &no_pre_mine,
                    &latest_submission,
                    &no_pre_mine_hour,
                    mode,
                    gpu_batch_size,
                    args.optimized,
                );
            }

            #[cfg(not(feature = "cuda"))]
            {
                solve_cpu_only(
                    &init_rom(&no_pre_mine),
                    &format!(
                        "{}{}{}{}{}{}",
                        address,
                        challenge_id,
                        difficulty,
                        no_pre_mine,
                        latest_submission,
                        no_pre_mine_hour
                    ),
                    u32::from_str_radix(&difficulty, 16).unwrap(),
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn solve(
    address: &str,
    challenge_id: &str,
    difficulty: &str,
    no_pre_mine: &str,
    latest_submission: &str,
    no_pre_mine_hour: &str,
    solver_mode: SolverMode,
    batch_size: usize,
    optimized: bool,
) {
    let rom = init_rom(no_pre_mine);
    let difficulty_mask = u32::from_str_radix(difficulty, 16).unwrap();

    let suffix = format!(
        "{}{}{}{}{}{}",
        address, challenge_id, difficulty, no_pre_mine, latest_submission, no_pre_mine_hour
    );

    match solver_mode {
        SolverMode::Cpu => {
            solve_cpu_only(&rom, &suffix, difficulty_mask);
        }
        SolverMode::Gpu => {
            if optimized {
                if let Some(nonce) =
                    solve_multi_gpu_optimized(&rom, &suffix, difficulty_mask, batch_size)
                {
                    println!("{:016x}", nonce);
                }
            } else {
                if let Some(nonce) = solve_multi_gpu(&rom, &suffix, difficulty_mask, batch_size) {
                    println!("{:016x}", nonce);
                }
            }
        }
        SolverMode::Auto => {
            let has_gpu = match gpu::CudaAshmaize::new() {
                Ok(_) => match gpu::CudaAshmaize::get_device_count() {
                    Ok(count) if count > 0 => true,
                    _ => false,
                },
                Err(_) => false,
            };

            if has_gpu {
                if optimized {
                    if let Some(nonce) =
                        solve_multi_gpu_optimized(&rom, &suffix, difficulty_mask, batch_size)
                    {
                        println!("{:016x}", nonce);
                    }
                } else {
                    if let Some(nonce) = solve_multi_gpu(&rom, &suffix, difficulty_mask, batch_size)
                    {
                        println!("{:016x}", nonce);
                    }
                }
            } else {
                eprintln!("No GPU available, falling back to CPU");
                solve_cpu_only(&rom, &suffix, difficulty_mask);
            }
        }
        SolverMode::Mixed => {
            if optimized {
                if let Some(nonce) =
                    solve_mixed_optimized(&rom, &suffix, difficulty_mask, batch_size)
                {
                    println!("{:016x}", nonce);
                }
            } else {
                if let Some(nonce) = solve_mixed(&rom, &suffix, difficulty_mask, batch_size) {
                    println!("{:016x}", nonce);
                }
            }
        }
    }
}

fn solve_cpu_only(rom: &Rom, suffix: &str, difficulty_mask: u32) {
    let rom = Arc::new(rom.clone());
    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));
    let start_nonce = 0;

    (0..NUM_THREADS).into_par_iter().for_each(|thread_id| {
        let rom = Arc::clone(&rom);
        let mut local_nonce = start_nonce + thread_id as u64;
        let stride = NUM_THREADS as u64;

        let mut preimage = String::with_capacity(16 + suffix.len());

        while !found.load(Ordering::Relaxed) {
            preimage.clear();
            use std::fmt::Write;
            write!(&mut preimage, "{:016x}{}", local_nonce, &suffix).unwrap();

            let hash_result = hash(preimage.as_bytes(), &rom, 8, 256);

            if hash_structure_good(&hash_result, difficulty_mask) {
                found.store(true, Ordering::Relaxed);
                result_nonce.store(local_nonce, Ordering::Relaxed);
                break;
            }

            local_nonce += stride;
        }
    });

    if found.load(Ordering::Relaxed) {
        println!("{:016x}", result_nonce.load(Ordering::Relaxed));
    }
}

#[cfg(feature = "cuda")]
fn solve_multi_gpu(
    rom: &Rom,
    suffix: &str,
    difficulty_mask: u32,
    batch_size: usize,
) -> Option<u64> {
    use std::sync::mpsc;
    use std::thread;

    let _init_cuda = match gpu::CudaAshmaize::new() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {:?}, falling back to CPU", e);
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let gpu_count = match gpu::CudaAshmaize::get_device_count() {
        Ok(count) if count > 0 => {
            eprintln!("Detected {} GPU(s)", count);
            count
        }
        _ => {
            eprintln!("No GPUs detected, falling back to CPU");
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let mut gpus = Vec::new();
    for gpu_id in 0..gpu_count {
        match gpu::CudaAshmaize::new_with_device(gpu_id) {
            Ok(c) => {
                eprintln!(
                    "GPU {}: {}",
                    gpu_id,
                    c.get_device_info()
                        .unwrap_or_else(|_| format!("GPU {}", gpu_id))
                );
                gpus.push(Arc::new(c));
            }
            Err(e) => {
                eprintln!(
                    "Failed to init GPU {}: {:?}, falling back to CPU",
                    gpu_id, e
                );
                solve_cpu_only(rom, suffix, difficulty_mask);
                return None;
            }
        }
    }

    let batch_per_gpu = batch_size;
    let rom = Arc::new(rom.clone());
    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));

    let (result_tx, result_rx) = mpsc::channel::<(usize, Result<Vec<[u8; 64]>, String>)>();

    let mut handles = Vec::new();
    for (gpu_id, cuda) in gpus.iter().enumerate() {
        let cuda = Arc::clone(cuda);
        let rom = Arc::clone(&rom);
        let suffix = suffix.to_string();
        let result_tx = result_tx.clone();
        let found = Arc::clone(&found);
        let result_nonce = Arc::clone(&result_nonce);

        let handle = thread::spawn(move || {
            let base_nonce = (gpu_id as u64) * (batch_per_gpu as u64);
            let mut local_nonce = base_nonce;

            while !found.load(Ordering::Relaxed) {
                let mut batch_salts = Vec::with_capacity(batch_per_gpu);
                for i in 0..batch_per_gpu {
                    let nonce = local_nonce + i as u64;
                    let preimage = format!("{:016x}{}", nonce, suffix);
                    batch_salts.push(preimage.into_bytes());
                }

                let salt_refs: Vec<&[u8]> = batch_salts.iter().map(|s| s.as_slice()).collect();

                let result = cuda
                    .hash_parallel(&salt_refs, &rom, 8, 256)
                    .map_err(|e| format!("{:?}", e));

                if let Ok(hashes) = &result {
                    for (i, hash) in hashes.iter().enumerate() {
                        if hash_structure_good(hash, difficulty_mask) {
                            let winning_nonce = local_nonce + i as u64;
                            found.store(true, Ordering::Relaxed);
                            result_nonce.store(winning_nonce, Ordering::Relaxed);
                            return;
                        }
                    }
                }

                result_tx.send((gpu_id, result)).ok();
                local_nonce += (batch_per_gpu * gpu_count) as u64;
            }
        });
        handles.push(handle);
    }
    drop(result_tx);

    let _collector = thread::spawn(move || {
        while let Ok((gpu_id, result)) = result_rx.recv() {
            if let Err(e) = result {
                eprintln!("GPU {} error: {}", gpu_id, e);
            }
        }
    });

    for handle in handles {
        handle.join().ok();
    }

    if found.load(Ordering::Relaxed) {
        Some(result_nonce.load(Ordering::Relaxed))
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn solve_multi_gpu_optimized(
    rom: &Rom,
    suffix: &str,
    difficulty_mask: u32,
    batch_size: usize,
) -> Option<u64> {
    use std::sync::mpsc;
    use std::thread;

    // Initialize CUDA first to ensure proper device detection
    let _init_cuda = match gpu::CudaAshmaize::new() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {:?}, falling back to CPU", e);
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let gpu_count = match gpu::CudaAshmaize::get_device_count() {
        Ok(count) if count > 0 => {
            eprintln!("Detected {} GPU(s)", count);
            count
        }
        _ => {
            eprintln!("No GPUs detected, falling back to CPU");
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let mut optimized_gpus = Vec::new();
    for gpu_id in 0..gpu_count {
        match gpu::CudaAshmaizeOptimized::new_with_device(gpu_id, rom, suffix, batch_size, 8, 256) {
            Ok(c) => {
                eprintln!(
                    "GPU {}: {}",
                    gpu_id,
                    c.get_device_info()
                        .unwrap_or_else(|_| format!("GPU {}", gpu_id))
                );
                optimized_gpus.push(c);
            }
            Err(e) => {
                eprintln!(
                    "Failed to init optimized GPU {}: {:?}, falling back to CPU",
                    gpu_id, e
                );
                solve_cpu_only(rom, suffix, difficulty_mask);
                return None;
            }
        }
    }

    let batch_per_gpu = batch_size;
    let rom = Arc::new(rom.clone());
    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));

    let (result_tx, result_rx) = mpsc::channel::<(usize, Result<(), String>)>();

    let mut handles = Vec::new();
    for (gpu_id, mut cuda) in optimized_gpus.into_iter().enumerate() {
        let rom = Arc::clone(&rom);
        let result_tx = result_tx.clone();
        let found = Arc::clone(&found);
        let result_nonce = Arc::clone(&result_nonce);

        let handle = thread::spawn(move || {
            let base_nonce = (gpu_id as u64) * (batch_per_gpu as u64);
            let mut local_nonce = base_nonce;

            while !found.load(Ordering::Relaxed) {
                let result = cuda
                    .hash_batch_optimized(local_nonce, batch_per_gpu, &rom, 8, 256)
                    .map_err(|e| format!("{:?}", e));

                match result {
                    Ok(hashes) => {
                        for (i, hash) in hashes.iter().enumerate() {
                            if hash_structure_good(hash, difficulty_mask) {
                                let winning_nonce = local_nonce + i as u64;
                                found.store(true, Ordering::Relaxed);
                                result_nonce.store(winning_nonce, Ordering::Relaxed);
                                return;
                            }
                        }
                        result_tx.send((gpu_id, Ok(()))).ok();
                    }
                    Err(e) => {
                        result_tx.send((gpu_id, Err(e))).ok();
                    }
                }

                local_nonce += (batch_per_gpu * gpu_count) as u64;
            }
        });
        handles.push(handle);
    }
    drop(result_tx);

    let _collector = thread::spawn(move || {
        while let Ok((gpu_id, result)) = result_rx.recv() {
            if let Err(e) = result {
                eprintln!("GPU {} error: {}", gpu_id, e);
            }
        }
    });

    for handle in handles {
        handle.join().ok();
    }

    if found.load(Ordering::Relaxed) {
        Some(result_nonce.load(Ordering::Relaxed))
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn solve_mixed_optimized(
    rom: &Rom,
    suffix: &str,
    difficulty_mask: u32,
    batch_size: usize,
) -> Option<u64> {
    use std::thread;

    let gpu_count = match gpu::CudaAshmaize::get_device_count() {
        Ok(count) if count > 0 => {
            eprintln!(
                "Mixed optimized mode: {} CPU threads + {} GPU(s)",
                NUM_THREADS, count
            );
            count
        }
        _ => {
            eprintln!("No GPUs available for mixed mode, falling back to CPU only");
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let mut optimized_gpus = Vec::new();
    for gpu_id in 0..gpu_count {
        match gpu::CudaAshmaizeOptimized::new_with_device(gpu_id, rom, suffix, batch_size, 8, 256) {
            Ok(c) => {
                optimized_gpus.push(c);
            }
            Err(e) => {
                eprintln!("Failed to init optimized GPU {}: {:?}", gpu_id, e);
            }
        }
    }

    if optimized_gpus.is_empty() {
        eprintln!("No GPUs initialized successfully, falling back to CPU only");
        solve_cpu_only(rom, suffix, difficulty_mask);
        return None;
    }

    let rom = Arc::new(rom.clone());
    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));

    // CPU workers
    let cpu_found = Arc::clone(&found);
    let cpu_result = Arc::clone(&result_nonce);
    let cpu_rom = Arc::clone(&rom);
    let cpu_suffix = suffix.to_string();
    let cpu_handle = thread::spawn(move || {
        (0..NUM_THREADS).into_par_iter().for_each(|thread_id| {
            let rom = Arc::clone(&cpu_rom);
            let mut local_nonce = thread_id as u64;
            let stride = NUM_THREADS as u64;
            let mut preimage = String::with_capacity(16 + cpu_suffix.len());

            while !cpu_found.load(Ordering::Relaxed) {
                preimage.clear();
                use std::fmt::Write;
                write!(&mut preimage, "{:016x}{}", local_nonce, &cpu_suffix).unwrap();

                let hash = hash(preimage.as_bytes(), &rom, 8, 256);
                if hash_structure_good(&hash, difficulty_mask) {
                    cpu_found.store(true, Ordering::Relaxed);
                    cpu_result.store(local_nonce, Ordering::Relaxed);
                    return;
                }
                local_nonce += stride;
            }
        });
    });

    // GPU workers
    let mut gpu_handles = Vec::new();
    for (gpu_id, mut cuda) in optimized_gpus.into_iter().enumerate() {
        let gpu_found = Arc::clone(&found);
        let gpu_result = Arc::clone(&result_nonce);
        let gpu_rom = Arc::clone(&rom);

        let gpu_handle = thread::spawn(move || {
            let base_nonce = (1000000 + gpu_id * 1000000) as u64;
            let mut local_nonce = base_nonce;

            while !gpu_found.load(Ordering::Relaxed) {
                match cuda.hash_batch_optimized(local_nonce, batch_size, &gpu_rom, 8, 256) {
                    Ok(hashes) => {
                        for (i, hash) in hashes.iter().enumerate() {
                            if hash_structure_good(hash, difficulty_mask) {
                                let winning_nonce = local_nonce + i as u64;
                                gpu_found.store(true, Ordering::Relaxed);
                                gpu_result.store(winning_nonce, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("GPU {} error: {:?}", gpu_id, e);
                        return;
                    }
                }
                local_nonce += batch_size as u64;
            }
        });
        gpu_handles.push(gpu_handle);
    }

    cpu_handle.join().ok();
    for handle in gpu_handles {
        handle.join().ok();
    }

    if found.load(Ordering::Relaxed) {
        Some(result_nonce.load(Ordering::Relaxed))
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn solve_mixed(rom: &Rom, suffix: &str, difficulty_mask: u32, batch_size: usize) -> Option<u64> {
    use std::thread;

    let _init_cuda = match gpu::CudaAshmaize::new() {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "Failed to initialize CUDA: {:?}, falling back to CPU only",
                e
            );
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    let gpu_count = match gpu::CudaAshmaize::get_device_count() {
        Ok(count) if count > 0 => count,
        _ => {
            eprintln!("No GPUs available for mixed mode, falling back to CPU only");
            solve_cpu_only(rom, suffix, difficulty_mask);
            return None;
        }
    };

    eprintln!(
        "Mixed mode: {} CPU threads + {} GPU(s)",
        NUM_THREADS, gpu_count
    );

    let mut gpus = Vec::new();
    for gpu_id in 0..gpu_count {
        match gpu::CudaAshmaize::new_with_device(gpu_id) {
            Ok(c) => gpus.push(Arc::new(c)),
            Err(e) => {
                eprintln!("Failed to init GPU {}: {:?}", gpu_id, e);
                return None;
            }
        }
    }

    let batch_per_gpu = batch_size;
    let rom = Arc::new(rom.clone());
    let found = Arc::new(AtomicBool::new(false));
    let result_nonce = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for (gpu_id, cuda) in gpus.iter().enumerate() {
        let cuda = Arc::clone(cuda);
        let rom = Arc::clone(&rom);
        let suffix = suffix.to_string();
        let found = Arc::clone(&found);
        let result_nonce = Arc::clone(&result_nonce);

        let handle = thread::spawn(move || {
            let base_nonce = (gpu_id as u64) * (batch_per_gpu as u64);
            let mut local_nonce = base_nonce;

            while !found.load(Ordering::Relaxed) {
                let mut batch_salts = Vec::with_capacity(batch_per_gpu);
                for i in 0..batch_per_gpu {
                    let nonce = local_nonce + i as u64;
                    let preimage = format!("{:016x}{}", nonce, suffix);
                    batch_salts.push(preimage.into_bytes());
                }

                let salt_refs: Vec<&[u8]> = batch_salts.iter().map(|s| s.as_slice()).collect();

                match cuda.hash_parallel(&salt_refs, &rom, 8, 256) {
                    Ok(hashes) => {
                        for (i, hash) in hashes.iter().enumerate() {
                            if hash_structure_good(hash, difficulty_mask) {
                                let winning_nonce = local_nonce + i as u64;
                                found.store(true, Ordering::Relaxed);
                                result_nonce.store(winning_nonce, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                    Err(_) => {}
                }

                local_nonce += (batch_per_gpu * gpu_count) as u64;
            }
        });
        handles.push(handle);
    }

    let cpu_found = Arc::clone(&found);
    let cpu_result = Arc::clone(&result_nonce);
    let rom_cpu = Arc::clone(&rom);
    let suffix_cpu = suffix.to_string();

    let cpu_handle = thread::spawn(move || {
        let start_nonce = (gpu_count as u64 + 1) * 1_000_000_000;

        (0..NUM_THREADS).into_par_iter().for_each(|thread_id| {
            let mut local_nonce = start_nonce + thread_id;
            let stride = NUM_THREADS;

            while !cpu_found.load(Ordering::Relaxed) {
                let preimage = format!("{:016x}{}", local_nonce, suffix_cpu);
                let hash_result = hash(preimage.as_bytes(), &rom_cpu, 8, 256);

                if hash_structure_good(&hash_result, difficulty_mask) {
                    cpu_found.store(true, Ordering::Relaxed);
                    cpu_result.store(local_nonce, Ordering::Relaxed);
                    break;
                }

                local_nonce += stride;
            }
        });
    });

    for handle in handles {
        handle.join().ok();
    }
    cpu_handle.join().ok();

    if found.load(Ordering::Relaxed) {
        Some(result_nonce.load(Ordering::Relaxed))
    } else {
        None
    }
}
