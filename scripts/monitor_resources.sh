#!/bin/bash
# Usage: ./unified_monitor.sh [update_interval_in_seconds]
UPDATE_INTERVAL=${1:-5}

while true; do
    clear
    echo "System Monitor - $(date)"
    echo "===================================================================="
    
    # Overall CPU usage using top.
    CPU_LINE=$(top -bn1 | grep "Cpu(s)")
    CPU_USAGE=$(echo "$CPU_LINE" | awk -F'id,' '{ split($1, cpu, " "); usage=100-cpu[length(cpu)]; printf "%.1f%%", usage }')
    
    # Memory usage (used/total) using free.
    MEM_USAGE=$(free -h | awk 'NR==2 {print $3 "/" $2}')
    
    # Disk usage for root filesystem (df).
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5 " used (" $3 " of " $2 ")"}')
    
    # GPU usage and memory usage using nvidia-smi (if available).
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        GPU_MEM_RAW=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
        GPU_MEM=$(echo "$GPU_MEM_RAW" | awk -F',' '{printf "%sMB / %sMB", $1, $2}')
        GPU_INFO="Util: ${GPU_UTIL}% | Mem: ${GPU_MEM}"
    else
        GPU_INFO="N/A"
    fi

    # Unified summary table.
    printf "%-20s: %-30s\n" "CPU Overall Usage" "$CPU_USAGE"
    printf "%-20s: %-30s\n" "Memory Usage" "$MEM_USAGE"
    printf "%-20s: %-30s\n" "Disk Usage (/)" "$DISK_USAGE"
    printf "%-20s: %-30s\n" "GPU Info" "$GPU_INFO"
    echo "===================================================================="
    
    # Per CPU core utilization using mpstat if available.
    if command -v mpstat >/dev/null 2>&1; then
        echo "Per CPU Core Utilization:"
        # Using mpstat over one second; skip headers and the "all" aggregate.
        mpstat -P ALL 1 1 | awk 'NR>4 && $2!="all" { usage = 100 - $NF; printf "    CPU%-3s: %.1f%%\n", $2, usage }'
    else
        echo "mpstat not available - cannot display per core utilization."
    fi
    
    sleep "$UPDATE_INTERVAL"
done