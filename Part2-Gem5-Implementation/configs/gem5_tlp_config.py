#!/usr/bin/env python3
"""
gem5 Configuration Script for FloatSimdFU Design Space Exploration
Explores different opLat/issueLat combinations (summing to 7 cycles)
across varying thread counts to analyze TLP performance.

Usage: python3 gem5_tlp_config.py --oplat <opLat> --issuelat <issueLat> --threads <num> --binary <path>
"""

import argparse
import sys
import os

# Import gem5 modules (these paths assume standard gem5 installation)
try:
    import m5
    from m5.objects import *
    from m5.util import addToPath
except ImportError:
    print("Error: gem5 Python modules not found. Run from gem5 directory or set PYTHONPATH.")
    sys.exit(1)


def create_minor_cpu_with_custom_fu(oplat, issuelat):
    """
    Create MinorCPU with custom FloatSimdFU parameters.
    
    Args:
        oplat: Operation latency (cycles for instruction execution)
        issuelat: Issue latency (cycles before next instruction can issue)
    
    Returns:
        Configured MinorCPU object
    """
    
    # Create custom functional unit pool based on MinorDefaultFUPool
    class CustomFloatSimdFU(MinorFU):
        # Configure FloatSimd operations with custom latencies
        opLat = oplat
        issueLat = issuelat
        # Specify operations this FU handles
        opClasses = minorMakeOpClassSet(['FloatAdd', 'FloatCmp', 'FloatCvt',
                                          'FloatMult', 'FloatMultAcc',
                                          'FloatDiv', 'FloatSqrt',
                                          'FloatMisc',
                                          'SimdAdd', 'SimdAddAcc', 'SimdAlu',
                                          'SimdCmp', 'SimdCvt', 'SimdMisc',
                                          'SimdMult', 'SimdMultAcc',
                                          'SimdShift', 'SimdShiftAcc',
                                          'SimdSqrt', 'SimdFloatAdd',
                                          'SimdFloatAlu', 'SimdFloatCmp',
                                          'SimdFloatCvt', 'SimdFloatDiv',
                                          'SimdFloatMisc', 'SimdFloatMult',
                                          'SimdFloatMultAcc', 'SimdFloatSqrt',
                                          'SimdReduceAdd', 'SimdReduceAlu',
                                          'SimdReduceCmp', 'SimdFloatReduceAdd',
                                          'SimdFloatReduceCmp'])
    
    class CustomFUPool(MinorFUPool):
        funcUnits = [
            MinorDefaultIntFU(), MinorDefaultIntFU(),
            MinorDefaultIntMulFU(), MinorDefaultIntDivFU(),
            MinorDefaultFloatSimdFU(),  # Keep default for comparison
            CustomFloatSimdFU(),  # Our custom FU
            MinorDefaultMemFU(), MinorDefaultMemFU(),
            MinorDefaultMiscFU(),
        ]
    
    cpu = MinorCPU(cpu_id=0)
    cpu.executeFuncUnits = CustomFUPool()
    
    return cpu


def build_system(args):
    """
    Build complete gem5 system with multi-core configuration.
    
    Args:
        args: Command-line arguments containing oplat, issuelat, threads, binary
    
    Returns:
        Configured System object
    """
    
    # Create system
    system = System()
    
    # Set clock and voltage
    system.clk_domain = SrcClockDomain()
    system.clk_domain.clock = '2GHz'
    system.clk_domain.voltage_domain = VoltageDomain()
    
    # Set memory mode and range
    system.mem_mode = 'timing'
    system.mem_ranges = [AddrRange('4GB')]
    
    # Create CPUs
    system.cpu = [create_minor_cpu_with_custom_fu(args.oplat, args.issuelat) 
                   for _ in range(args.threads)]
    
    # Create memory bus
    system.membus = SystemXBar()
    
    # Create cache hierarchy (L1 I/D caches)
    for cpu in system.cpu:
        cpu.icache = L1ICache(size='32kB', assoc=4)
        cpu.dcache = L1DCache(size='32kB', assoc=4)
        
        cpu.icache.cpu_side = cpu.icache_port
        cpu.dcache.cpu_side = cpu.dcache_port
        
        cpu.icache.mem_side = system.membus.cpu_side_ports
        cpu.dcache.mem_side = system.membus.cpu_side_ports
        
        # Connect interrupt ports (required for SE mode)
        cpu.createInterruptController()
    
    # Create memory controller
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports
    
    # Connect system port
    system.system_port = system.membus.cpu_side_ports
    
    # Set up workload (SE mode)
    if not os.path.exists(args.binary):
        print(f"Error: Binary '{args.binary}' not found!")
        sys.exit(1)
    
    # Create process for each CPU
    for i, cpu in enumerate(system.cpu):
        process = Process()
        process.cmd = [args.binary, str(args.vector_size), str(args.threads)]
        cpu.workload = process
        cpu.createThreads()
    
    # Create root and instantiate
    root = Root(full_system=False, system=system)
    m5.instantiate()
    
    return root


def main():
    parser = argparse.ArgumentParser(
        description='gem5 TLP exploration with configurable FloatSimdFU'
    )
    parser.add_argument('--oplat', type=int, required=True,
                        help='Operation latency (cycles)')
    parser.add_argument('--issuelat', type=int, required=True,
                        help='Issue latency (cycles)')
    parser.add_argument('--threads', type=int, default=2,
                        help='Number of threads/cores (default: 2)')
    parser.add_argument('--binary', type=str, required=True,
                        help='Path to daxpy binary')
    parser.add_argument('--vector-size', type=int, default=10000,
                        help='DAXPY vector size (default: 10000)')
    parser.add_argument('--output-dir', type=str, default='m5out',
                        help='Output directory for stats (default: m5out)')
    
    args = parser.parse_args()
    
    # Validate latency sum
    if args.oplat + args.issuelat != 7:
        print(f"Warning: opLat ({args.oplat}) + issueLat ({args.issuelat}) != 7")
    
    # Validate parameters
    if args.oplat < 1 or args.issuelat < 1:
        print("Error: Both opLat and issueLat must be >= 1")
        sys.exit(1)
    
    print(f"=== gem5 TLP Configuration ===")
    print(f"FloatSimdFU: opLat={args.oplat}, issueLat={args.issuelat}")
    print(f"Threads: {args.threads}")
    print(f"Vector size: {args.vector_size}")
    print(f"Binary: {args.binary}")
    print(f"Output: {args.output_dir}")
    print("=" * 40)
    
    # Set output directory
    m5.options.outdir = args.output_dir
    
    # Build and run simulation
    root = build_system(args)
    
    print("Starting simulation...")
    exit_event = m5.simulate()
    
    print(f"Simulation completed: {exit_event.getCause()}")
    print(f"Simulated ticks: {m5.curTick()}")
    print(f"Statistics written to: {args.output_dir}/stats.txt")


if __name__ == '__main__':
    main()
