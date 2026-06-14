#!/bin/bash
# SkillNav one-shot launcher: starts the full 4-terminal pipeline inside one tmux session.
#
# Layout (tmux session: skillnav)
#   window 0:  rviz   |  planner
#              -------+----------
#                     habitat
#
# Order matches README: rviz brings up roscore, then planner, then Habitat.
# VLM servers run in a SEPARATE tmux session ("vlm_servers") and are started
# by scripts/start_vlm_servers.sh if not already up.
#
# Usage:
#   ./scripts/run_skillnav.sh                  # exploration_multiagent.launch + hm3dv2
#   ./scripts/run_skillnav.sh --single         # exploration.launch (single-agent, no VLM agents)
#   ./scripts/run_skillnav.sh --dataset mp3d   # pass dataset to habitat_evaluation.py
#   ./scripts/run_skillnav.sh --no-vlm         # skip starting VLM servers
#   ./scripts/run_skillnav.sh --attach         # attach after starting

set -u

SKILLNAV_DIR="/home/yangz/Nav/SkillNav"
CONDA_ENV="skillnav"
TMUX_SESSION="skillnav"

LAUNCH_FILE="exploration_multiagent.launch"
DATASET="hm3dv2"
START_VLM=1
ATTACH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)    LAUNCH_FILE="exploration.launch"; shift ;;
        --dataset)   DATASET="$2"; shift 2 ;;
        --no-vlm)    START_VLM=0; shift ;;
        --attach)    ATTACH=1; shift ;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session '$TMUX_SESSION' already exists. Run scripts/cleanup.sh first, or:"
    echo "  tmux attach -t $TMUX_SESSION"
    exit 1
fi

# 1. VLM servers (separate tmux session, idempotent).
if [[ $START_VLM -eq 1 ]]; then
    if tmux has-session -t vlm_servers 2>/dev/null; then
        echo "[run] VLM servers already running (tmux: vlm_servers) — skipping start."
    else
        echo "[run] Starting VLM servers (tmux: vlm_servers)..."
        "$SKILLNAV_DIR/scripts/start_vlm_servers.sh" start >/dev/null
    fi
fi

# Per-pane preamble. Order matters:
#   1) cd into workspace
#   2) conda activate (so habitat_evaluation.py has its deps)
#   3) source ROS noetic + this workspace's devel (so roslaunch sees our pkgs)
PREAMBLE="cd $SKILLNAV_DIR \
&& source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV \
&& source /opt/ros/noetic/setup.bash && source $SKILLNAV_DIR/devel/setup.bash"

echo "[run] Creating tmux session: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 60 -n main

# 3-pane layout: top-left rviz, top-right planner, bottom habitat.
tmux split-window -h -t "$TMUX_SESSION:0.0"
tmux split-window -v -t "$TMUX_SESSION:0.0"

# Pane 0: RViz (also starts roscore).
tmux select-pane -t "$TMUX_SESSION:0.0" -T "rviz+roscore"
tmux send-keys   -t "$TMUX_SESSION:0.0" "$PREAMBLE && roslaunch exploration_manager rviz.launch" Enter

# Pane 1 (top-right): planner — wait briefly for roscore to come up.
tmux select-pane -t "$TMUX_SESSION:0.2" -T "planner"
tmux send-keys   -t "$TMUX_SESSION:0.2" \
    "$PREAMBLE && until rostopic list >/dev/null 2>&1; do sleep 1; done && roslaunch exploration_manager $LAUNCH_FILE" Enter

# Pane 2 (bottom-left): habitat — wait for planner to be subscribing.
# HDF5_DISABLE_VERSION_CHECK=1 silences the h5py 1.10.4 vs 1.12.2 header/library
# mismatch in the skillnav conda env, which otherwise aborts at import time.
tmux select-pane -t "$TMUX_SESSION:0.1" -T "habitat"
tmux send-keys   -t "$TMUX_SESSION:0.1" \
    "$PREAMBLE && export HDF5_DISABLE_VERSION_CHECK=1 && until rostopic info /habitat/odom 2>/dev/null | grep -q Subscribers; do sleep 1; done && sleep 3 && python habitat_evaluation.py --dataset $DATASET" Enter

tmux select-pane -t "$TMUX_SESSION:0.0"

echo "[run] Started. Layout:"
echo "    ┌──────────────┬──────────────┐"
echo "    │ rviz+roscore │ planner      │"
echo "    │              │ ($LAUNCH_FILE)"
echo "    ├──────────────┴──────────────┤"
echo "    │ habitat ($DATASET)          │"
echo "    └─────────────────────────────┘"
echo
echo "  Attach:   tmux attach -t $TMUX_SESSION"
echo "  Stop:     scripts/cleanup.sh        (add --all to also kill roscore + VLM)"

if [[ $ATTACH -eq 1 ]]; then
    exec tmux attach -t "$TMUX_SESSION"
fi
