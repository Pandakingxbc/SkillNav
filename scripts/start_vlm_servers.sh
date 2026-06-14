#!/bin/bash

# VLM Servers Startup Script for SkillNav
# Usage: ./start_vlm_servers.sh [start|stop|restart|status]
#
# This script manages VLM servers in tmux sessions using the skillnav conda environment.

set -e

# Configuration
SESSION_NAME="vlm_servers"
CONDA_ENV="skillnav"
SKILLNAV_DIR="/home/yangz/Nav/SkillNav"

# Server configuration: name:port:module
SERVERS=(
    "grounding_dino:12181:vlm.detector.grounding_dino"
    "blip2_itm:12182:vlm.itm.blip2itm"
    "sam:12183:vlm.segmentor.sam"
    "yolov7:12184:vlm.detector.yolov7"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if tmux is installed
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        echo -e "${RED}Error: tmux is not installed. Please install tmux first.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install tmux"
        exit 1
    fi
}

# Check if conda environment exists
check_conda_env() {
    source ~/miniconda3/etc/profile.d/conda.sh
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        echo -e "${RED}Error: Conda environment '$CONDA_ENV' not found.${NC}"
        echo "Available environments:"
        conda env list
        exit 1
    fi
}

# Start all VLM servers
start_servers() {
    check_tmux
    check_conda_env

    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$SESSION_NAME' already exists. Use 'restart' to restart servers.${NC}"
        return 1
    fi

    echo -e "${BLUE}Creating tmux session: $SESSION_NAME${NC}"
    echo -e "${BLUE}Using conda environment: $CONDA_ENV${NC}"
    echo -e "${BLUE}Working directory: $SKILLNAV_DIR${NC}"

    # Create tmux session
    tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

    # Create 4 panes (2x2 grid)
    tmux split-window -h -t "$SESSION_NAME"
    tmux split-window -v -t "$SESSION_NAME:0.0"
    tmux split-window -v -t "$SESSION_NAME:0.2"

    # Start each server in a different pane
    local pane_index=0
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r server_name port module <<< "$server_config"

        echo -e "${BLUE}Starting $server_name on port $port in pane $pane_index...${NC}"

        # Set pane title and start server
        tmux select-pane -t "$SESSION_NAME:0.$pane_index" -T "$server_name:$port"
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "echo '=== $server_name (port $port) ==='" Enter
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "cd $SKILLNAV_DIR" Enter
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV" Enter
        sleep 0.5
        tmux send-keys -t "$SESSION_NAME:0.$pane_index" "python -m $module --port $port" Enter

        sleep 1
        pane_index=$((pane_index + 1))
    done

    # Select the first pane
    tmux select-pane -t "$SESSION_NAME:0.0"

    echo -e "${GREEN}All VLM servers have been started!${NC}"
    echo ""
    echo -e "${BLUE}Server URLs:${NC}"
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r server_name port module <<< "$server_config"
        echo -e "  $server_name: http://localhost:$port"
    done
    echo ""
    echo -e "${BLUE}Pane Layout (2x2):${NC}"
    echo -e "  ┌─────────────────┬─────────────────┐"
    echo -e "  │ grounding_dino  │ sam             │"
    echo -e "  │ (12181)         │ (12183)         │"
    echo -e "  ├─────────────────┼─────────────────┤"
    echo -e "  │ blip2_itm       │ yolov7          │"
    echo -e "  │ (12182)         │ (12184)         │"
    echo -e "  └─────────────────┴─────────────────┘"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  Attach to session:  ${GREEN}tmux attach -t $SESSION_NAME${NC}"
    echo -e "  Stop servers:       ${GREEN}$0 stop${NC}"
    echo -e "  Check status:       ${GREEN}$0 status${NC}"
    echo ""
    echo -e "${YELLOW}tmux tips:${NC}"
    echo -e "  Switch panes: Ctrl+b then arrow keys"
    echo -e "  Detach:       Ctrl+b then d"
}

# Stop all servers
stop_servers() {
    check_tmux

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$SESSION_NAME' is not running.${NC}"
        return 0
    fi

    echo -e "${BLUE}Stopping all VLM servers...${NC}"
    tmux kill-session -t "$SESSION_NAME"

    echo -e "${GREEN}All VLM servers have been stopped.${NC}"
}

# Restart all servers
restart_servers() {
    echo -e "${BLUE}Restarting VLM servers...${NC}"
    stop_servers
    sleep 2
    start_servers
}

# Check status of servers
status_servers() {
    check_tmux

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${RED}Session '$SESSION_NAME' is not running.${NC}"
        return 1
    fi

    echo -e "${GREEN}Session '$SESSION_NAME' is running.${NC}"
    echo ""

    # Check each port
    echo -e "${BLUE}Checking server ports:${NC}"
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r server_name port module <<< "$server_config"
        if curl -s --connect-timeout 2 "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "  $server_name (port $port): ${GREEN}RUNNING${NC}"
        else
            echo -e "  $server_name (port $port): ${YELLOW}LOADING or NOT READY${NC}"
        fi
    done

    echo ""
    echo -e "${BLUE}To attach to the session: tmux attach -t $SESSION_NAME${NC}"
}

# Main logic
case "${1:-start}" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    restart)
        restart_servers
        ;;
    status)
        status_servers
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all VLM servers (default)"
        echo "  stop    - Stop all VLM servers"
        echo "  restart - Restart all VLM servers"
        echo "  status  - Show status of VLM servers"
        exit 1
        ;;
esac
