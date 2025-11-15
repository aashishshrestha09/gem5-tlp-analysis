#!/usr/bin/env python3
"""
gem5 Statistics Parser and Analysis Tool
Extracts and analyzes performance metrics from gem5 simulation outputs
for FloatSimdFU design space exploration.

Metrics extracted:
- Simulation ticks (time)
- IPC (instructions per cycle) per core
- CPI (cycles per instruction) per core
- FloatSimdFU utilization
- Cache miss rates
- Speedup vs. single-threaded baseline

Usage: python3 analyze_results.py [--results-dir <path>] [--baseline-dir <path>]
"""

import os
import re
import argparse
import csv
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_stats_file(stats_path):
    """
    Parse gem5 stats.txt file and extract key metrics.
    
    Args:
        stats_path: Path to stats.txt file
    
    Returns:
        Dictionary of parsed statistics
    """
    stats = {
        'sim_ticks': 0,
        'sim_seconds': 0,
        'num_cores': 0,
        'ipc': [],
        'cpi': [],
        'committed_insts': [],
        'num_cycles': [],
        'fu_utilization': {},
        'cache_stats': {}
    }
    
    if not os.path.exists(stats_path):
        print(f"Warning: Stats file not found: {stats_path}")
        return None
    
    with open(stats_path, 'r') as f:
        content = f.read()
        
        # Extract simulation time
        match = re.search(r'simTicks\s+(\d+)', content)
        if match:
            stats['sim_ticks'] = int(match.group(1))
        
        match = re.search(r'simSeconds\s+([\d.]+)', content)
        if match:
            stats['sim_seconds'] = float(match.group(1))
        
        # Extract per-core stats
        core_pattern = r'system\.cpu(\d+)\.committedInsts\s+(\d+)'
        for match in re.finditer(core_pattern, content):
            core_id = int(match.group(1))
            insts = int(match.group(2))
            stats['committed_insts'].append(insts)
        
        stats['num_cores'] = len(stats['committed_insts'])
        
        # Extract IPC/CPI
        ipc_pattern = r'system\.cpu(\d+)\.ipc\s+([\d.]+)'
        for match in re.finditer(ipc_pattern, content):
            ipc = float(match.group(2))
            stats['ipc'].append(ipc)
        
        cpi_pattern = r'system\.cpu(\d+)\.cpi\s+([\d.]+)'
        for match in re.finditer(cpi_pattern, content):
            cpi = float(match.group(2))
            stats['cpi'].append(cpi)
        
        # Extract cache miss rates
        cache_patterns = {
            'l1d_miss_rate': r'system\.cpu\d+\.dcache\.overallMissRate::total\s+([\d.]+)',
            'l1i_miss_rate': r'system\.cpu\d+\.icache\.overallMissRate::total\s+([\d.]+)',
        }
        
        for key, pattern in cache_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                stats['cache_stats'][key] = [float(m) for m in matches]
    
    return stats


def extract_config_from_path(result_path):
    """
    Extract configuration parameters from result directory name.
    Expected format: oplat<N>_issuelat<M>_t<T>
    
    Returns:
        Dictionary with oplat, issuelat, threads
    """
    dirname = os.path.basename(result_path)
    match = re.match(r'oplat(\d+)_issuelat(\d+)_t(\d+)', dirname)
    
    if match:
        return {
            'oplat': int(match.group(1)),
            'issuelat': int(match.group(2)),
            'threads': int(match.group(3))
        }
    return None


def analyze_results(results_dir, baseline_dir=None):
    """
    Analyze all simulation results and generate summary.
    
    Args:
        results_dir: Directory containing simulation outputs
        baseline_dir: Optional baseline single-threaded result for speedup calculation
    
    Returns:
        DataFrame with aggregated results
    """
    results = []
    baseline_ticks = None
    
    # Load baseline if provided
    if baseline_dir and os.path.exists(baseline_dir):
        baseline_stats = parse_stats_file(os.path.join(baseline_dir, 'stats.txt'))
        if baseline_stats:
            baseline_ticks = baseline_stats['sim_ticks']
            print(f"Baseline simulation time: {baseline_ticks} ticks")
    
    # Process all result directories
    for subdir in sorted(os.listdir(results_dir)):
        result_path = os.path.join(results_dir, subdir)
        
        if not os.path.isdir(result_path):
            continue
        
        config = extract_config_from_path(result_path)
        if not config:
            continue
        
        stats_path = os.path.join(result_path, 'stats.txt')
        stats = parse_stats_file(stats_path)
        
        if not stats or not stats['ipc']:
            print(f"Warning: Incomplete stats for {subdir}")
            continue
        
        # Calculate aggregate metrics
        avg_ipc = np.mean(stats['ipc']) if stats['ipc'] else 0
        avg_cpi = np.mean(stats['cpi']) if stats['cpi'] else 0
        total_insts = sum(stats['committed_insts'])
        
        # Calculate speedup
        speedup = None
        if baseline_ticks and stats['sim_ticks'] > 0:
            speedup = baseline_ticks / stats['sim_ticks']
        
        # Parallel efficiency
        efficiency = (speedup / config['threads'] * 100) if speedup else None
        
        result_entry = {
            'opLat': config['oplat'],
            'issueLat': config['issuelat'],
            'threads': config['threads'],
            'sim_ticks': stats['sim_ticks'],
            'sim_seconds': stats['sim_seconds'],
            'avg_ipc': avg_ipc,
            'avg_cpi': avg_cpi,
            'total_instructions': total_insts,
            'speedup': speedup,
            'efficiency_%': efficiency,
            'min_ipc': min(stats['ipc']) if stats['ipc'] else 0,
            'max_ipc': max(stats['ipc']) if stats['ipc'] else 0,
        }
        
        results.append(result_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Sort by configuration
        df = df.sort_values(['threads', 'opLat'])
    
    return df


def generate_plots(df, output_dir='figures'):
    """
    Generate visualization plots from analysis results.
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Color scheme
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    # Plot 1: IPC vs opLat for different thread counts
    plt.figure(figsize=(10, 6))
    for threads in sorted(df['threads'].unique()):
        subset = df[df['threads'] == threads]
        plt.plot(subset['opLat'], subset['avg_ipc'], 
                marker='o', label=f'{threads} threads', linewidth=2)
    
    plt.xlabel('Operation Latency (opLat)', fontsize=12)
    plt.ylabel('Average IPC', fontsize=12)
    plt.title('IPC vs. Operation Latency for Different Thread Counts', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ipc_vs_oplat.png'), dpi=300)
    print(f"Saved: {output_dir}/ipc_vs_oplat.png")
    plt.close()
    
    # Plot 2: Speedup vs threads for different opLat configurations
    if 'speedup' in df.columns and df['speedup'].notna().any():
        plt.figure(figsize=(10, 6))
        for oplat in sorted(df['opLat'].unique()):
            subset = df[df['opLat'] == oplat]
            if not subset.empty:
                plt.plot(subset['threads'], subset['speedup'], 
                        marker='s', label=f'opLat={oplat}', linewidth=2)
        
        # Add ideal speedup line
        max_threads = df['threads'].max()
        plt.plot([1, max_threads], [1, max_threads], 
                'k--', label='Ideal', alpha=0.5)
        
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title('Parallel Speedup vs. Thread Count', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_vs_threads.png'), dpi=300)
        print(f"Saved: {output_dir}/speedup_vs_threads.png")
        plt.close()
    
    # Plot 3: Heatmap of execution time (sim_ticks)
    plt.figure(figsize=(10, 6))
    pivot = df.pivot_table(values='sim_ticks', index='threads', columns='opLat')
    im = plt.imshow(pivot.values, aspect='auto', cmap='viridis')
    
    plt.colorbar(im, label='Simulation Ticks')
    plt.xlabel('Operation Latency (opLat)', fontsize=12)
    plt.ylabel('Number of Threads', fontsize=12)
    plt.title('Execution Time Heatmap', fontsize=14)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_execution_time.png'), dpi=300)
    print(f"Saved: {output_dir}/heatmap_execution_time.png")
    plt.close()
    
    # Plot 4: Parallel efficiency
    if 'efficiency_%' in df.columns and df['efficiency_%'].notna().any():
        plt.figure(figsize=(10, 6))
        for oplat in sorted(df['opLat'].unique()):
            subset = df[df['opLat'] == oplat]
            if not subset.empty and subset['efficiency_%'].notna().any():
                plt.plot(subset['threads'], subset['efficiency_%'], 
                        marker='^', label=f'opLat={oplat}', linewidth=2)
        
        plt.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Parallel Efficiency (%)', fontsize=12)
        plt.title('Parallel Efficiency vs. Thread Count', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_vs_threads.png'), dpi=300)
        print(f"Saved: {output_dir}/efficiency_vs_threads.png")
        plt.close()


def save_summary(df, output_path='results/summary.csv'):
    """Save analysis summary to CSV."""
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nSummary saved to: {output_path}")


def print_summary_table(df):
    """Print formatted summary table to console."""
    if df.empty:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    # Format numeric columns
    display_cols = ['opLat', 'issueLat', 'threads', 'avg_ipc', 'avg_cpi', 
                    'speedup', 'efficiency_%', 'sim_seconds']
    
    display_df = df[display_cols].copy()
    
    pd.options.display.float_format = '{:.4f}'.format
    print(display_df.to_string(index=False))
    print("="*80)
    
    # Find best configurations
    if 'speedup' in df.columns and df['speedup'].notna().any():
        for threads in sorted(df['threads'].unique()):
            subset = df[df['threads'] == threads]
            if not subset.empty:
                best = subset.loc[subset['speedup'].idxmax()]
                print(f"\nBest config for {threads} threads: "
                      f"opLat={best['opLat']}, issueLat={best['issueLat']}, "
                      f"speedup={best['speedup']:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze gem5 TLP simulation results'
    )
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing simulation results')
    parser.add_argument('--baseline-dir', type=str, default=None,
                        help='Baseline single-threaded result directory for speedup')
    parser.add_argument('--output-csv', type=str, default='results/summary.csv',
                        help='Output CSV file for summary')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Directory to save generated figures')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("=== gem5 TLP Results Analysis ===\n")
    
    # Analyze results
    df = analyze_results(args.results_dir, args.baseline_dir)
    
    if df.empty:
        print("No valid results found!")
        return 1
    
    # Print summary
    print_summary_table(df)
    
    # Save CSV
    save_summary(df, args.output_csv)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(df, args.figures_dir)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    exit(main())
