FROM osrf/ros:humble-desktop-full

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

SHELL [ "/bin/bash" , "-c" ]
RUN apt-get update
RUN apt-get install -y git && apt-get install -y python3-pip
RUN source /opt/ros/humble/setup.bash 
    # && cd ~/ \
    # && git clone https://github.com/VictoriousMango/Vision-Navigation.git
RUN echo "All Done !!!"