#!/bin/bash
# Chạy COLMAP + train Scene-GS cho ĐÚNG 1 scene. Dùng nội bộ bởi run_all_scenes.sh
# (chạy song song nhiều scene qua xargs -P), có thể tự gọi riêng lẻ nếu muốn.
#
# Cách gọi: bash scripts/run_one_scene.sh "<tên_scene>|<đường_dẫn_ảnh>"

set -e
cd "$(dirname "$0")/.."

IFS='|' read -r name images_dir <<< "$1"

export PYTHONNOUSERSITE=1
export DISPLAY=:0
PYTHON=/home/ml4u/conda_envs/safe-gs/bin/python

workspace_dir="outputs/workspace/${name}"
ckpt="${workspace_dir}/checkpoints/scene_gs_30000.ply"
log_file="${workspace_dir}/run.log"
mkdir -p "$workspace_dir"

if [ -f "$ckpt" ]; then
    echo "[SKIP] ${name} đã có checkpoint 30000, bỏ qua."
    exit 0
fi

{
    echo "======================================================================"
    echo ">>> SCENE: ${name}  ($(date))"
    echo "======================================================================"

    if [ ! -d "${workspace_dir}/sfm/sparse/0" ]; then
        echo "--- [${name}] Chạy COLMAP (SfM) ---"
        "$PYTHON" scripts/01_run_perception.py \
            --images_dir "$images_dir" \
            --workspace_dir "$workspace_dir" \
            --skip_seg
    else
        echo "[SKIP] Đã có output SfM, bỏ qua COLMAP."
    fi

    echo "--- [${name}] Train Scene-GS (30000 iterations) ---"
    "$PYTHON" scripts/02_run_scene_gs.py \
        --sfm_dir "${workspace_dir}/sfm/sparse/0" \
        --raw_image_dir "$images_dir" \
        --workspace_dir "$workspace_dir"

    echo ">>> XONG SCENE: ${name}  ($(date))"
} > "$log_file" 2>&1

echo "[DONE] ${name} -> xem log tại ${log_file}"
