#!/usr/bin/env bash
# Linux equivalent of run_loop.ps1
# Usage: bash run_loop.sh [N]   (default N=60)

N=${1:-60}
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Parse edge_device list from config.yaml
EDGE_DEVICES=$(python3 -c "
import yaml, sys
with open('$DIR/config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
devs = cfg.get('cpcdrl', {}).get('edge_device', [])
if isinstance(devs, str): devs = [devs]
print(' '.join(devs))
")

CLOUD_DEVICES=$(python3 -c "
import yaml
with open('$DIR/config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
devs = cfg.get('cpcdrl', {}).get('cloud_device', [])
if isinstance(devs, str): devs = [devs]
print(' '.join(devs))
")

echo -e "\033[36mEdge devices : $EDGE_DEVICES\033[0m"
echo -e "\033[36mCloud devices: $CLOUD_DEVICES\033[0m"
echo -e "\033[36mRunning $N episodes. Python: $PYTHON\033[0m"

for i in $(seq 1 $N); do
    echo -e "\n\033[36m=== Episode $i / $N ===\033[0m"

    $PYTHON "$DIR/server.py" &
    SRV_PID=$!
    sleep 2

    PIDS=()
    for dev in $EDGE_DEVICES; do
        $PYTHON "$DIR/client.py" --layer_id 1 --edge_device "$dev" &
        PIDS+=($!)
    done

    LAYER_ID=2
    for dev in $CLOUD_DEVICES; do
        $PYTHON "$DIR/client.py" --layer_id $LAYER_ID &
        PIDS+=($!)
        LAYER_ID=$((LAYER_ID + 1))
    done

    wait $SRV_PID
    for pid in "${PIDS[@]}"; do wait $pid; done

    echo -e "\033[32mEpisode $i done.\033[0m"
done

echo -e "\n\033[32mAll $N episodes finished.\033[0m"
