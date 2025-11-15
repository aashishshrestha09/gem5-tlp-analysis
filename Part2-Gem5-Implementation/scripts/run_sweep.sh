#!/bin/bash
#
# Automated sweep script for FloatSimdFU design space exploration
# Runs gem5 simulations for all opLat/issueLat combinations summing to 7
# across multiple thread counts (2, 4, 8 threads)
#

set -e  # Exit on error

# Configuration
GEM5_BUILD="build/X86/gem5.opt"  # Adjust to your gem5 build
CONFIG_SCRIPT="configs/gem5_tlp_config.py"
BINARY="src/daxpy_mt"
VECTOR_SIZE=50000
THREAD_COUNTS=(2 4 8)
BASE_OUTPUT_DIR="results"

# Check if gem5 executable exists
if [ ! -f "$GEM5_BUILD" ]; then
    echo "Error: gem5 executable not found at $GEM5_BUILD"
    echo "Please build gem5 or update GEM5_BUILD variable"
    exit 1
fi

# Check if config script exists
if [ ! -f "$CONFIG_SCRIPT" ]; then
    echo "Error: Config script not found at $CONFIG_SCRIPT"
    exit 1
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: DAXPY binary not found at $BINARY"
    echo "Please compile with: gcc -O2 -pthread src/daxpy_mt.c -o src/daxpy_mt"
    exit 1
fi

# Create results directory
mkdir -p "$BASE_OUTPUT_DIR"

# Generate all opLat/issueLat combinations that sum to 7
echo "=== FloatSimdFU Design Space Exploration ==="
echo "Exploring opLat + issueLat = 7 combinations"
echo "Thread counts: ${THREAD_COUNTS[@]}"
echo "Vector size: $VECTOR_SIZE"
echo ""

total_runs=0
successful_runs=0

# Loop through latency combinations
for oplat in {1..6}; do
    issuelat=$((7 - oplat))
    
    # Loop through thread counts
    for threads in "${THREAD_COUNTS[@]}"; do
        output_dir="${BASE_OUTPUT_DIR}/oplat${oplat}_issuelat${issuelat}_t${threads}"
        
        echo "-------------------------------------------"
        echo "Running: opLat=$oplat, issueLat=$issuelat, threads=$threads"
        echo "Output: $output_dir"
        
        total_runs=$((total_runs + 1))
        
        # Run gem5 simulation
        if $GEM5_BUILD \
            --outdir="$output_dir" \
            "$CONFIG_SCRIPT" \
            --oplat "$oplat" \
            --issuelat "$issuelat" \
            --threads "$threads" \
            --vector-size "$VECTOR_SIZE" \
            --binary "$BINARY" \
            > "${output_dir}/simulation.log" 2>&1; then
            
            echo "✓ Success"
            successful_runs=$((successful_runs + 1))
        else
            echo "✗ Failed (see ${output_dir}/simulation.log)"
        fi
    done
done

echo ""
echo "=== Sweep Complete ==="
echo "Total runs: $total_runs"
echo "Successful: $successful_runs"
echo "Failed: $((total_runs - successful_runs))"
echo ""
echo "Results stored in: $BASE_OUTPUT_DIR/"
echo "Run analysis script: python3 scripts/analyze_results.py"
