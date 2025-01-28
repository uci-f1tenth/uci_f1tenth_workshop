# F1Tenth @ UCI Workshop Repo

<p align="center">
  <img src="./etc/f1t_uci_logo.png" width="450" title="hover text">
</p>

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
