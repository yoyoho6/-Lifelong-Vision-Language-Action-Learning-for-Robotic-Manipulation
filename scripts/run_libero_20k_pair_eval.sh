#!/usr/bin/env bash
set -euo pipefail

repo_dir="/media/ubuntu/data/home/hy/copy/openpi-main"
server_python="$repo_dir/.venv/bin/python"
sim_python="$repo_dir/examples/libero/.venv/bin/python"
server_script="$repo_dir/scripts/serve_policy.py"
sim_script="$repo_dir/examples/libero/main.py"
config_name="pi0_libero_low_mem_finetune"
port=18000

log_root="$repo_dir/logs/libero_eval_20k_20260806"
video_root="$repo_dir/data/libero/eval_20k_20260806"
mkdir -p "$log_root" "$video_root"

export PYTHONPATH="${PYTHONPATH:-}:$repo_dir/third_party/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

server_pid=""

stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}

trap stop_server EXIT INT TERM

wait_for_server() {
    local server_log="$1"
    local ready=0
    for _ in $(seq 1 900); do
        local server_state
        server_state="$(ps -o stat= -p "$server_pid" 2>/dev/null || true)"
        if [[ -z "$server_state" || "$server_state" == Z* ]]; then
            set +e
            wait "$server_pid"
            local server_status=$?
            set -e
            echo "Policy server exited during startup with status $server_status."
            tail -n 80 "$server_log"
            return 1
        fi
        if grep -q "server listening on .*:$port" "$server_log" && \
            "$sim_python" -c "import socket; s=socket.create_connection(('127.0.0.1', $port), timeout=1); s.close()" \
            >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Policy server did not become ready within 900 seconds."
        tail -n 80 "$server_log"
        return 1
    fi
}

run_eval() {
    local label="$1"
    local checkpoint_dir="$2"
    shift 2
    local task_ids=("$@")
    local server_log="$log_root/${label}_server.log"
    local eval_log="$log_root/${label}_eval.log"
    local video_dir="$video_root/$label"
    mkdir -p "$video_dir"

    echo "Starting $label policy server from $checkpoint_dir"
    CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false "$server_python" "$server_script" \
        --port "$port" \
        policy:checkpoint \
        --policy.config "$config_name" \
        --policy.dir "$checkpoint_dir" \
        >"$server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$server_log"

    echo "Running $label evaluation in task order: ${task_ids[*]}"
    "$sim_python" "$sim_script" \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-suite-name libero_spatial \
        --args.eval-task-ids "${task_ids[@]}" \
        --args.num-trials-per-task 50 \
        --args.video-out-path "$video_dir" \
        >"$eval_log" 2>&1

    stop_server
    echo "Completed $label evaluation"
}

if [[ "${ONLY_GPU2:-0}" != "1" ]]; then
    run_eval \
        gpu0_seed43_step20000_first2_retry3_spatial \
        "$repo_dir/checkpoints/pi0_libero_low_mem_finetune/final_replay_seed43_ascending_nodebuginfs_20260805/20000" \
        6 4
fi

if [[ "${ONLY_GPU0:-0}" != "1" ]]; then
    run_eval \
        gpu2_seed42_step20000_first2_retry3_spatial \
        "$repo_dir/checkpoints/pi0_libero_low_mem_finetune/final_replay_seed42_descending_nodebuginfs_20260805/20000" \
        9 2
fi
