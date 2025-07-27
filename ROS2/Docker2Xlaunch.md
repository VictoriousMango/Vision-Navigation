### Making GUI Enabled Container:
```bash
docker run -d -t --name CV_Nav_VSLAM -e DISPLAY=host.docker.internal:0.0 -it c810de0c9dec
docker run -d -t --name <Container Name> -e DISPLAY=host.docker.internal:0.0 -e LIBGL_ALWAYS_INDIRECT=0 --runtime=nvidia <Image Identifier> bash 
```
Run XLaunch
see the following command for testing GUI:
```bash
 ros2 run turtlesim turtlesim_node
```