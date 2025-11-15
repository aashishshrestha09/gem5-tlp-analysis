# gem5 Thread-Level Parallelism Analysis

[![gem5](https://img.shields.io/badge/gem5-simulator-blue)](https://www.gem5.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)

## Overview

Performance analysis tool for investigating Thread-Level Parallelism using the gem5 architectural simulator. Analyzes FloatSimdFU latency configurations on multi-threaded DAXPY workloads.

### Features

- Automated parameter sweep across multiple latency configurations
- Multi-threaded DAXPY benchmark with configurable thread counts
- Python-based analysis and visualization pipeline
- Comprehensive performance metrics (IPC, CPI, speedup, cache statistics)

## Project Structure

```text
Assignment-6/
├── Part2-Gem5-Implementation/
│   ├── configs/
│   │   └── gem5_tlp_config.py         # gem5 configuration script
│   ├── src/
│   │   ├── daxpy_mt.c                 # Multi-threaded DAXPY benchmark
│   │   └── daxpy_mt                   # Compiled binary
│   ├── scripts/
│   │   ├── run_sweep.sh               # Automation for 18 simulations
│   │   ├── analyze_results.py         # Performance analysis
│   │   └── generate_speedup_chart.py  # Chart generation
│   ├── results/                       # 18 simulation outputs
│   │   └── summary.csv
│   ├── figures/                       # Performance visualizations (PNG)
│   ├── Makefile
│   └── requirements.txt
├── .gitignore
└── README.MD
```

## Components

### Simulation Configuration

- gem5 MinorCPU model (4-stage in-order pipeline, 2GHz)
- Configurable FloatSimdFU latency parameters
- Multi-core system with private L1 and shared L2 caches

### Benchmark

- Multi-threaded DAXPY implementation (y = a*x + y)
- POSIX threads with configurable thread counts
- Workload designed to stress floating-point execution units

### Analysis Tools

- Automated simulation parameter sweeps
- Performance metric extraction and aggregation
- Visualization generation (IPC charts, heatmaps, speedup analysis)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- GCC with pthread support
- Python packages: `matplotlib`, `pandas`, `numpy`

### Installation

```bash
# Navigate to project directory
cd Part2-Gem5-Implementation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the DAXPY benchmark
make build
```

### Running Analysis

```bash
# Analyze simulation results and generate visualizations
python scripts/analyze_results.py

# Generate speedup charts
python scripts/generate_speedup_chart.py
```

### Output

- `results/summary.csv` - Performance metrics across all configurations
- `figures/*.png` - Visualization charts (IPC, heatmaps, speedup analysis)

## Performance Analysis

Analyzes trade-offs in FloatSimdFU design:

- Operation latency vs. issue latency impact on throughput
- Thread scalability and parallel efficiency
- Cache behavior under varying thread counts
- Pipeline utilization and stall patterns

Results include detailed visualizations and quantitative metrics for architecture optimization.

## Repository Structure

```text
Part2-Gem5-Implementation/
├── configs/          # gem5 simulation configurations
├── src/              # Source code and compiled binaries
├── scripts/          # Analysis and automation scripts
├── results/          # Simulation outputs
└── figures/          # Performance visualizations
```

