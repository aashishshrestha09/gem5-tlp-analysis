#!/usr/bin/env python3
"""
Generate bar chart showing parallel speedup across all configurations.
Uses data from Table 2 in the implementation report.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 2: Speedup values for each configuration
# Format: (opLat, issueLat): {2: speedup_2thread, 4: speedup_4thread, 8: speedup_8thread}
speedup_data = {
    (1, 6): {2: 1.85, 4: 3.42, 8: 5.89},
    (2, 5): {2: 1.83, 4: 3.38, 8: 5.76},
    (3, 4): {2: 1.81, 4: 3.35, 8: 5.64},
    (4, 3): {2: 1.79, 4: 3.31, 8: 5.52},
    (5, 2): {2: 1.77, 4: 3.28, 8: 5.41},
    (6, 1): {2: 1.75, 4: 3.24, 8: 5.30},
}

# Prepare data for grouped bar chart
latency_configs = ['(1,6)', '(2,5)', '(3,4)', '(4,3)', '(5,2)', '(6,1)']
thread_counts = [2, 4, 8]
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Extract speedup values
speedups_2t = [speedup_data[k][2] for k in [(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)]]
speedups_4t = [speedup_data[k][4] for k in [(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)]]
speedups_8t = [speedup_data[k][8] for k in [(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)]]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(latency_configs))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, speedups_2t, width, label='2 Threads', color=colors[0], alpha=0.8)
bars2 = ax.bar(x, speedups_4t, width, label='4 Threads', color=colors[1], alpha=0.8)
bars3 = ax.bar(x + width, speedups_8t, width, label='8 Threads', color=colors[2], alpha=0.8)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# Formatting
ax.set_xlabel('Latency Configuration (opLat, issueLat)', fontsize=12, fontweight='bold')
ax.set_ylabel('Parallel Speedup', fontsize=12, fontweight='bold')
ax.set_title('Parallel Speedup Across Thread Counts and FloatSimdFU Latencies', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(latency_configs)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 7)

# Add reference line for ideal speedup
ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=4, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=8, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('../figures/speedup_bar_chart.png', dpi=300, bbox_inches='tight')
print("Speedup bar chart saved to: ../figures/speedup_bar_chart.png")
