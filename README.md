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
git clone https://github.com/<your-user>/ur5e_ai.git
