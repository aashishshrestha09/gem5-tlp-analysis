# Makefile for Assignment 6 - TLP Exploration with gem5

# Configuration
CC = gcc
CFLAGS = -O2 -pthread -Wall
SRC_DIR = src
BUILD_DIR = build
BINARY = $(SRC_DIR)/daxpy_mt
VECTOR_SIZE = 50000

# gem5 configuration (update these paths)
GEM5_ROOT = $(HOME)/gem5
GEM5_BUILD = $(GEM5_ROOT)/build/X86/gem5.opt
GEM5_CONFIG = configs/gem5_tlp_config.py

# Output directories
RESULTS_DIR = results
FIGURES_DIR = figures

.PHONY: all build test clean sweep analyze help

all: build

help:
	@echo "Assignment 6 - TLP Exploration Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  build       - Compile the DAXPY multi-threaded kernel"
	@echo "  test        - Run local test of DAXPY (outside gem5)"
	@echo "  sweep       - Run full parameter sweep in gem5"
	@echo "  analyze     - Parse results and generate plots"
	@echo "  clean       - Remove build artifacts and results"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  GEM5_ROOT   = $(GEM5_ROOT)"
	@echo "  GEM5_BUILD  = $(GEM5_BUILD)"
	@echo "  VECTOR_SIZE = $(VECTOR_SIZE)"

build: $(BINARY)
	@echo "✓ Build complete: $(BINARY)"

$(BINARY): $(SRC_DIR)/daxpy_mt.c
	@echo "Compiling DAXPY multi-threaded kernel..."
	$(CC) $(CFLAGS) -o $@ $< -lm
	@echo "✓ Compilation successful"

test: $(BINARY)
	@echo "Running local test (2 threads, 10000 elements)..."
	@./$(BINARY) 10000 2
	@echo ""
	@echo "Running local test (4 threads, 10000 elements)..."
	@./$(BINARY) 10000 4

check-gem5:
	@if [ ! -f "$(GEM5_BUILD)" ]; then \
		echo "Error: gem5 not found at $(GEM5_BUILD)"; \
		echo "Update GEM5_ROOT in Makefile or build gem5"; \
		exit 1; \
	fi
	@echo "✓ gem5 found: $(GEM5_BUILD)"

sweep: build check-gem5
	@echo "Starting parameter sweep..."
	@chmod +x scripts/run_sweep.sh
	@./scripts/run_sweep.sh

analyze:
	@echo "Analyzing results and generating plots..."
	@python3 scripts/analyze_results.py \
		--results-dir $(RESULTS_DIR) \
		--output-csv $(RESULTS_DIR)/summary.csv \
		--figures-dir $(FIGURES_DIR)
	@echo "✓ Analysis complete"
	@echo "  Summary: $(RESULTS_DIR)/summary.csv"
	@echo "  Figures: $(FIGURES_DIR)/"

clean:
	@echo "Cleaning build artifacts and results..."
	rm -f $(BINARY)
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(FIGURES_DIR)/*
	@echo "✓ Clean complete"

# Quick single-run target for testing
quick-test: build check-gem5
	@echo "Running quick gem5 test (opLat=3, issueLat=4, 2 threads)..."
	@mkdir -p $(RESULTS_DIR)/quick_test
	$(GEM5_BUILD) \
		--outdir=$(RESULTS_DIR)/quick_test \
		$(GEM5_CONFIG) \
		--oplat 3 \
		--issuelat 4 \
		--threads 2 \
		--vector-size 10000 \
		--binary $(BINARY)
	@echo "✓ Quick test complete. Results in: $(RESULTS_DIR)/quick_test/"
