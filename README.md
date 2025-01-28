# F1Tenth @ UCI Workshop Repo

<p align="center">
  <img src="./etc/f1t_uci_logo.png" width="450" title="hover text">
</p>

# Introduction
We add [DreamerV3](https://arxiv.org/abs/2301.04104) implementations of World Models into the F1tenth ROS2 environment in PyTorch. Referenced [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), [racing_dreamer](https://github.com/CPS-TUWien/racing_dreamer), and [dreamerv3](https://github.com/danijar/dreamerv3) repositories for the base code.

## How Has This Been Tested?

- [x] implemented f1tenth gym for testing environment.
- [ ] waiting on issac sim implementation for better environment.

## Types of changes

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)

## Checklist:

- [x] basic structure implementation(bare bones)
- [ ] LiDAR feed for encoded latent state
- [ ] World Model implementation
- [ ] Actor-Critic implementation
- [ ] Scaling reward system


# Set Up
Go into the gym directory you installed the program in.
```
cd uci-f1tenth-workshop/f1tenth_gym_ros
```

Run shell script:
```
sh scripts/run.sh
```
Open a web browser and go to link
```
http://localhost:8080/vnc.html
```
Run the ROS within the VM hosted in the docker simulator and run
```
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
```
Launch the simulation
```
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
