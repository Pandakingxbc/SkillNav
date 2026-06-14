#!/bin/bash
# SkillNav cleanup: kill leftover planner / habitat / ROS / VLM-server processes.
#
# Usage:
#   ./scripts/cleanup.sh            # planner + habitat + rviz  (keep roscore, keep VLM servers)
#   ./scripts/cleanup.sh --all      # everything above + roscore/rosmaster + VLM tmux session
#   ./scripts/cleanup.sh --ros      # also kill roscore/rosmaster (keep VLM servers)
#   ./scripts/cleanup.sh --vlm      # also kill the vlm_servers tmux session

set -u

KILL_ROS=0
KILL_VLM=0
for arg in "$@"; do
    case "$arg" in
        --all) KILL_ROS=1; KILL_VLM=1 ;;
        --ros) KILL_ROS=1 ;;
        --vlm) KILL_VLM=1 ;;
        -h|--help)
            sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# pkill helper: report what was killed (or that nothing matched), never fail the script.
kill_pat() {
    local label="$1" pat="$2"
    if pgrep -f "$pat" >/dev/null 2>&1; then
        pkill -f "$pat" 2>/dev/null || true
        echo "  killed: $label"
    fi
}

echo "[cleanup] Stopping SkillNav planner + habitat processes..."
kill_pat "roslaunch exploration_*"        "roslaunch.*exploration"
kill_pat "exploration_node"               "exploration_node"
kill_pat "tsp_solver"                     "lkh_mtsp_solver/tsp_node"
kill_pat "vlm_safe_agent_node.py"         "vlm_safe_agent_node.py"
kill_pat "vlm_memory_agent_node.py"       "vlm_memory_agent_node.py"
kill_pat "habitat_evaluation.py"          "habitat_evaluation.py"
kill_pat "rviz"                           "[r]viz"

if [[ $KILL_ROS -eq 1 ]]; then
    echo "[cleanup] Stopping ROS master..."
    kill_pat "rosmaster"  "rosmaster"
    kill_pat "roscore"    "[r]oscore"
    kill_pat "rosout"     "rosout"
fi

if [[ $KILL_VLM -eq 1 ]]; then
    echo "[cleanup] Stopping VLM tmux session..."
    if command -v tmux >/dev/null && tmux has-session -t vlm_servers 2>/dev/null; then
        tmux kill-session -t vlm_servers
        echo "  killed: tmux session vlm_servers"
    fi
fi

# Reap zombies whose parent is this shell tree.
sleep 0.3
wait 2>/dev/null || true

echo "[cleanup] Done."
