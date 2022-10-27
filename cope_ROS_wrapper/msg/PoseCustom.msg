string object_to_locate  # optional , if unspecified, returns everything found
---
std_msgs/Header header
geometry_msgs/Pose[] object_poses
string[] object_types
float32[] confidences
sensor_msgs/Image color_image  # real color image used for the prediction
sensor_msgs/Image depth_image  # real depth image used for the prediction
sensor_msgs/CameraInfo camera_info
---
string placeholder_feedback
