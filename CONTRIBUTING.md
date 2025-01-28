# Contributing to UCI_F1TENTH

This repository is intended for academic and research purposes at the University of California, Irvine. All contributions must adhere to applicable copyright laws, licensing agreements, and UCI’s academic integrity policies. Unauthorized use, distribution, or modification of this code in violation of these guidelines is strictly prohibited. By contributing, you agree to respect the intellectual property of others and ensure your work complies with academic use standards.

## Setup Instructions for ROS gym environment.
To use the NoVNC script for remote visualization of your simulations, follow these steps:

Clone the repository using SSH(HTTP will not grant you access for direct commits)
```bash
git clone "git@github.com:uci-f1tenth/uci_f1tenth_workshop.git"
```

Go into the gym directory you installed the program in.
```bash
cd uci-f1tenth-workshop/f1tenth_gym_ros
```

Run shell script:
```bash
sh scripts/run.sh
```

Open a web browser and go to link
```bash
http://localhost:8080/vnc.html
```

Run the ROS within the VM hosted in the docker simulator and run
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
```

Launch the simulation
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

## Building a node
*To build a node in ROS2, you must be within the vm root.
If you are not in the machine either run the script again, or run:
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Source two environments under /sim_ws/:
```bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
```
Run the build command under with your node name in setup.py
```bash
colcon build --packages-select your_node
```
Source the workspace environment again
```bash
source install/setup.bash
```
Run the node with the executable
```bash
ros2 run your_package your_executable
```

## Example execution of the dreamer node
```bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
colcon build --packages-select dreamer_node
source install/setup.bash
ros2 run agents dreamer_agent
```
---

## Guidelines for Contributions

### Commit Messages
- Write clear, concise commit messages that explain **what** was changed and **why**.
- Use the following format for commits:
  ```
  [TYPE] Brief summary of the changes

  Detailed explanation of the change (if needed).
  
  Fixes: #IssueNumber (if applicable)
  ```
  Examples of `TYPE`:
  - **feat**: Adding a new feature
  - **fix**: Fixing a bug
  - **refactor**: Code restructuring without changing functionality
  - **docs**: Documentation updates
  - **test**: Adding or modifying tests
  
### Code Style
- Follow **PEP 8** standards for Python code.
- Use meaningful variable and function names.
- Keep functions modular and focused on a single task.
- Add type hints where applicable for better code clarity.

### Pull Requests
1. Fork the repository and create a new branch for your feature or fix.
2. Ensure your branch is up-to-date with the `main` branch.
3. Submit a pull request with a clear description of your changes.
4. Address any comments or requested changes from reviewers promptly.

### Testing
- Ensure your code passes all tests before submitting a pull request.
- Add new tests if your changes introduce new functionality.

---

## Coding Policies

1. **Readability**: Prioritize readable, well-documented code over complex one-liners.
2. **Efficiency**: Optimize for computational efficiency without sacrificing clarity.
3. **Testing**: Write tests for all critical components to ensure robustness.
4. **Documentation**: Update documentation for any significant changes or additions.
5. **Collaboration**: Be respectful and constructive in code reviews and discussions.

---

Thank you for contributing to the F1TENTH DreamerV3 Project! Together, we can push the boundaries of autonomous racing and reinforcement learning.

