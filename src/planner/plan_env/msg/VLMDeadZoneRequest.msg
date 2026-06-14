# VLM Dead Zone Analysis Request
# Sent from C++ SafeAgent to Python VLM node

Header header

# Robot state when stuck
float64 robot_x
float64 robot_y
float64 robot_yaw              # heading in radians

# Escape attempt info
int32 escape_attempt_count     # number of failed escape attempts
int32 tryout_angles_tried      # how many angles were tried (0-7)

# Target topology node to potentially penalize
int32 target_node_id           # nearest topology node ID (-1 if none)
float64 target_node_x          # node position for reference
float64 target_node_y

# Navigation context
string target_description      # what we were trying to reach (optional)

# Image is sent via separate sensor_msgs/Image topic to avoid large message
