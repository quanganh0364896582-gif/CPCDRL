#!/usr/bin/env bash
# Linux equivalent of train_devices.ps1
# Usage: bash train_devices.sh [N] [DEVICE_4 DEVICE_7 ...]
# Default: N=200, devices=DEVICE_4 DEVICE_7

N=${1:-200}
shift || true
DEVICES=("${@:-DEVICE_4 DEVICE_7}")
if [ ${#DEVICES[@]} -eq 0 ]; then
    DEVICES=("DEVICE_4" "DEVICE_7")
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$DIR/config.yaml"

set_config_field() {
    local key="$1" value="$2"
    sed -i "s|^\(\s*${key}\s*:\s*\).*|\1${value}|" "$CONFIG"
}

for device in "${DEVICES[@]}"; do
    echo -e "\n\033[33m========================================\033[0m"
    echo -e "\033[33m  Training $device  ($N episodes)\033[0m"
    echo -e "\033[33m========================================\033[0m"

    set_config_field "max_frames" "160  # limit frames per episode for faster RL training (0 = no limit)"
    python3 -c "
import re, sys
with open('$CONFIG', encoding='utf-8') as f:
    content = f.read()
content = re.sub(r'edge_device\s*:\s*\[[^\]]+\]', 'edge_device:  [\"$device\"]', content)
with open('$CONFIG', 'w', encoding='utf-8') as f:
    f.write(content)
"
    echo -e "\033[36mConfig updated: max_frames=160, edge_device=$device\033[0m"

    bash "$DIR/run_loop.sh" $N

    echo -e "\n\033[32m[OK] $device training complete.\033[0m"
done

# Reset max_frames after all training
set_config_field "max_frames" "0  # limit frames per episode for faster RL training (0 = no limit)"
echo -e "\n\033[32mAll devices trained. max_frames reset to 0.\033[0m"
