# ur5e_ai – DMP control of UR5e in ROS 2 / Gazebo

This package implements DMPs and related
controllers to drive a UR5e manipulator in simulation and compare them
against a MoveIt-based baseline. The main demo executes a lemniscate
end-effector trajectory.

## 1. Prerequisites

- Ubuntu 22.04 + ROS 2 Humble
- MoveIt 2 for UR robots
- A ROS 2 workspace, e.g. `~/pai_ws`

## 2. Workspace setup

Create a workspace and clone the required packages into `src`:

```bash
# Create workspace
mkdir -p ~/pai_ws/src
cd ~/pai_ws/src

# Universal Robots description and Gazebo simulation
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation.git

# This package
git clone git@github.com:Zer0vowl/PAI_5ECTS_UR5e_ai_Flavian_Annus.git

# 2. Build workspace
cd ~/pai_ws
colcon build --symlink-install
source install/setup.bash

# Run UR5e simulation
cd ~/pai_ws
source install/setup.bash

ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py

# In a separate terminal run DMP or Moveit controller
cd ~/pai_ws
source install/setup.bash

ros2 run ur5e_ai ai_dmp_controller
ros2 run ur5e_ai ai_dmp_moveit_controller
