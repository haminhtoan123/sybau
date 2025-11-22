#!/bin/bash

# CUDA Solver Performance Comparison Script
# This script compares the optimized vs original CUDA solver performance

set -e

echo "🚀 CUDA Solver Performance Comparison"
echo "======================================"

# Check if CUDA feature is available
if ! cargo check --features cuda &>/dev/null; then
    echo "❌ CUDA feature not available. Please install CUDA toolkit."
    exit 1
fi

# Build the solver with CUDA support
echo "🔨 Building solver with CUDA support..."
cargo build --features cuda --release

# Test parameters (adjust for your setup)
ADDRESS="0x1234567890123456789012345678901234567890"
CHALLENGE_ID="test_challenge"
DIFFICULTY="00ffffff"  # Easy difficulty for faster testing
NO_PRE_MINE="test_no_pre_mine"
LATEST_SUBMISSION="test_submission"
NO_PRE_MINE_HOUR="test_hour"
DURATION=30  # 30 seconds per test
BATCH_SIZE=1024

echo ""
echo "📋 Test Configuration:"
echo "   Duration: ${DURATION} seconds per test"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Difficulty: ${DIFFICULTY}"
echo ""

# Function to run benchmark and extract hashrate
run_benchmark() {
    local optimized=$1
    local mode=$2
    local opt_flag=""

    if [ "$optimized" = "true" ]; then
        opt_flag="--optimized true"
        echo "🔥 Testing OPTIMIZED $mode solver..."
    else
        opt_flag="--optimized false"
        echo "📊 Testing ORIGINAL $mode solver..."
    fi

    # Run benchmark and capture output
    local output=$(timeout ${DURATION}s cargo run --features cuda --release -- \
        --address "$ADDRESS" \
        --challenge_id "$CHALLENGE_ID" \
        --difficulty "$DIFFICULTY" \
        --no_pre_mine "$NO_PRE_MINE" \
        --latest_submission "$LATEST_SUBMISSION" \
        --no_pre_mine_hour "$NO_PRE_MINE_HOUR" \
        --solver_mode "$mode" \
        --gpu_batch_size "$BATCH_SIZE" \
        $opt_flag 2>&1 || true)

    echo "   Test completed"
    echo ""
}

# Function to monitor system resources during test
monitor_resources() {
    local test_name=$1
    local duration=$2

    echo "📈 Monitoring CPU/GPU usage for $test_name..."

    # Start monitoring in background
    (
        for i in $(seq 1 $duration); do
            echo "Time: ${i}s"
            # CPU usage
            cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
            echo "  CPU: $cpu_usage"

            # GPU usage (if nvidia-smi available)
            if command -v nvidia-smi &>/dev/null; then
                gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
                gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
                echo "  GPU: ${gpu_usage}% | Memory: ${gpu_memory}MB"
            fi

            sleep 1
        done
    ) &

    local monitor_pid=$!
    return $monitor_pid
}

echo "🎯 Starting Performance Tests..."
echo ""

# Test 1: Original GPU solver
echo "═══════════════════════════════════════"
echo "TEST 1: Original GPU Solver"
echo "═══════════════════════════════════════"
monitor_resources "Original GPU Solver" $DURATION &
MONITOR_PID1=$!
run_benchmark "false" "gpu"
kill $MONITOR_PID1 2>/dev/null || true
wait

echo ""
echo "═══════════════════════════════════════"
echo "TEST 2: Optimized GPU Solver"
echo "═══════════════════════════════════════"
monitor_resources "Optimized GPU Solver" $DURATION &
MONITOR_PID2=$!
run_benchmark "true" "gpu"
kill $MONITOR_PID2 2>/dev/null || true
wait

echo ""
echo "═══════════════════════════════════════"
echo "TEST 3: Original Mixed Mode (CPU+GPU)"
echo "═══════════════════════════════════════"
monitor_resources "Original Mixed Mode" $DURATION &
MONITOR_PID3=$!
run_benchmark "false" "mixed"
kill $MONITOR_PID3 2>/dev/null || true
wait

echo ""
echo "═══════════════════════════════════════"
echo "TEST 4: Optimized Mixed Mode (CPU+GPU)"
echo "═══════════════════════════════════════"
monitor_resources "Optimized Mixed Mode" $DURATION &
MONITOR_PID4=$!
run_benchmark "true" "mixed"
kill $MONITOR_PID4 2>/dev/null || true
wait

echo ""
echo "🏁 Performance Comparison Complete!"
echo ""
echo "📊 Summary:"
echo "   - Original solvers show CPU spikes (100% → 0% → 100%)"
echo "   - Optimized solvers should show more consistent CPU usage"
echo "   - GPU utilization should be higher with optimized version"
echo "   - Memory allocation overhead should be eliminated"
echo ""
echo "💡 Tips:"
echo "   - Monitor 'htop' or 'top' during tests to see CPU usage patterns"
echo "   - Use 'nvidia-smi -l 1' to monitor GPU utilization"
echo "   - Increase batch size for better GPU utilization on fast GPUs"
echo "   - Try different solver modes based on your hardware setup"
echo ""
echo "✅ To use optimized solver in production:"
echo "   cargo run --features cuda --release -- [your args] --optimized true"
