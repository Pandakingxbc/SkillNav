#!/bin/bash
# Habitat evaluation runner with proper environment setup
# This script resolves the HDF5 version conflict between ROS and Python packages

set -e

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skillnav

# Source ROS setup
source /home/yangz/Nav/SkillNav/devel/setup.bash

# Use h5py's bundled HDF5 libraries
# h5py 3.7.0 uses HDF5 1.12.2 (libhdf5-fc7245dc.so.200.2.0)
H5PY_LIBS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/h5py.libs"

# Set LD_LIBRARY_PATH to include h5py's bundled libs FIRST
export LD_LIBRARY_PATH=$H5PY_LIBS_DIR:/opt/ros/noetic/lib:$LD_LIBRARY_PATH

# Find and preload the correct HDF5 library
HDF5_LIB=$(ls $H5PY_LIBS_DIR/libhdf5-*.so.200* 2>/dev/null | head -1)
HDF5_HL_LIB=$(ls $H5PY_LIBS_DIR/libhdf5_hl-*.so.200* 2>/dev/null | head -1)

if [ -f "$HDF5_LIB" ] && [ -f "$HDF5_HL_LIB" ]; then
    export LD_PRELOAD="$HDF5_LIB:$HDF5_HL_LIB"
    echo "Preloading HDF5: $HDF5_LIB"
fi

# Run the evaluation
exec python habitat_evaluation.py "$@"
