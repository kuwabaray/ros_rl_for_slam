#!/usr/bin/env bash

set -ex

export PYTHONPATH="$PYTHONPATH:/home/yukiya/catkin_ws/src/dqn_for_slam"

python3 -m dqn_for_slam.rl_worker
