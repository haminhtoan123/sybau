#!/bin/bash

# Test script to determine the correct flag format for Rust solver --optimized flag

echo "🧪 Testing Rust Solver Flag Formats"
echo "===================================="

SOLVER="./target/release/ashmaize-solver"
TEST_ARGS="--address addr1test --challenge-id test --difficulty 0000FFFF --no-pre-mine test --latest-submission test --no-pre-mine-hour test --solver-mode cpu"

if [ ! -f "$SOLVER" ]; then
    echo "❌ Rust solver not found at $SOLVER"
    echo "Please build first: cargo build --features cuda --release"
    exit 1
fi

echo "Testing different --optimized flag formats..."
echo ""

# Test 1: No optimized flag (should use default)
echo "1. Testing: No --optimized flag (default behavior)"
echo "Command: $SOLVER $TEST_ARGS"
timeout 5s $SOLVER $TEST_ARGS 2>&1 | head -3
echo ""

# Test 2: --optimized (flag only, no value)
echo "2. Testing: --optimized (flag only)"
echo "Command: $SOLVER $TEST_ARGS --optimized"
timeout 5s $SOLVER $TEST_ARGS --optimized 2>&1 | head -3
echo ""

# Test 3: --optimized true
echo "3. Testing: --optimized true (with value)"
echo "Command: $SOLVER $TEST_ARGS --optimized true"
timeout 5s $SOLVER $TEST_ARGS --optimized true 2>&1 | head -3
echo ""

# Test 4: --optimized=true
echo "4. Testing: --optimized=true (equals format)"
echo "Command: $SOLVER $TEST_ARGS --optimized=true"
timeout 5s $SOLVER $TEST_ARGS --optimized=true 2>&1 | head -3
echo ""

# Test 5: --optimized false
echo "5. Testing: --optimized false (disable)"
echo "Command: $SOLVER $TEST_ARGS --optimized false"
timeout 5s $SOLVER $TEST_ARGS --optimized false 2>&1 | head -3
echo ""

# Test 6: --optimized=false
echo "6. Testing: --optimized=false (equals format)"
echo "Command: $SOLVER $TEST_ARGS --optimized=false"
timeout 5s $SOLVER $TEST_ARGS --optimized=false 2>&1 | head -3