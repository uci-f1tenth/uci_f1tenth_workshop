# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:foxy

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
    nano \
    vim \
    python3-pip \
    libeigen3-dev \
    tmux \
    ros-foxy-rviz2
RUN apt-get -y dist-upgrade

RUN pip3 install transforms3d
RUN pip3 install torch
RUN pip3 install matplotlib
RUN pip3 install ruamel.yaml
RUN pip3 install einops
RUN pip3 install protobuf
RUN pip3 install gym

RUN pip3 install \
    gymnasium==1.0.0 \
    cloudpickle==3.1.1 \
    farama-notifications==0.0.4 \
    typing-extensions==4.12.2 \
    pettingzoo==1.24.3 \
    pybullet==3.2.7 \
    scipy==1.10.1 \
    yamldataclassconfig==1.5.0 \
    dataclasses-json==0.6.7

RUN pip3 install \
    markupsafe==2.1.5 \
    packaging==24.2 \
    absl-py==2.1.0 \
    grpcio==1.70.0 \
    tensorboard==2.14.0 \
    tensorboard-data-server==0.7.2 \
    werkzeug==3.0.6 \
    filelock==3.16.1 \
    fsspec==2025.2.0 \
    jinja2==3.1.5 \
    networkx==3.1 \
    sympy==1.13.1 \
    mpmath==1.3.0 \
    --no-deps  # Prevent conflicts with original packages

# f1tenth gym
RUN git clone https://github.com/f1tenth/f1tenth_gym
RUN cd f1tenth_gym && \
    pip3 install -e .

# racecar gym
RUN git clone https://github.com/axelbr/racecar_gym && \
    cd racecar_gym && \
    pip install -e .

# Ensure correct numpy version
RUN pip3 uninstall -y numpy
RUN pip3 install --no-cache-dir --ignore-installed numpy==1.23.0

# ros2 gym bridge
RUN mkdir -p sim_ws/src/f1tenth_gym_ros
COPY . /sim_ws/src/f1tenth_gym_ros
RUN source /opt/ros/foxy/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro foxy -y && \
    colcon build

RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
RUN echo "source install/local_setup.bash" >> ~/.bashrc

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]