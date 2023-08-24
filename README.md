# R-LIOM
related code for R-LIOM. The current version contains the code to convert and save the pseudo-visual image, and other modules of the algorithm will be updated gradually.

## Dependencies
1. Same dependencies as FAST-LIO.
2. GTSAM==4.0.3 and OpenCV==3.2.0, Other versions may work, but we have not tested.

## Run
```bash
catkin_make
source devel/setup.bash # in bash
# Modify the absolute paths in cfg/virtualCamParam.yaml and src/brute_sliding_intensity_map.cpp
rosrun image_generate brute_sliding_intensity_map
# run odometry algorithm such as FAST-LIO
# check the images of saved/imgs/ and saved/inpaint
```

## Acknowledgments
Thanks for LIO-SAM and FAST-LIO.