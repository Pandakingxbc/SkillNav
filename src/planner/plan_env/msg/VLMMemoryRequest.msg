# VLM Memory Agent Request
# Sent from C++ MemoryAgent to Python VLM node
# For creating/updating semantic memory at topology nodes

Header header

# Request type
uint8 REQUEST_CREATE = 0       # Create new memory for node
uint8 REQUEST_UPDATE = 1       # Update existing memory
uint8 REQUEST_VERIFY = 2       # Verify object presence (uses ObjectMap)
uint8 request_type

# Robot state
float64 robot_x
float64 robot_y
float64 robot_yaw              # heading in radians

# Target topology node
int32 target_node_id           # topology node to store memory
float64 target_node_x
float64 target_node_y

# Navigation context
string target_object           # current ObjectNav target (e.g., "toilet")
int32 target_cluster_id        # ObjectMap cluster ID being navigated to (-1 if none)

# Nearby node memories (for context in verification)
# Format: "node_id:room_type:objects;node_id:room_type:objects;..."
string nearby_memories

# Activation info
float64 semantic_value         # current semantic value at this position
int32 calls_this_minute        # how many VLM calls in last minute (quota tracking)

# Image is sent via /habitat/camera_rgb topic (habitat-to-ros native RGB)
# ObjectMap cluster image sent via /memory_agent/target_cluster_image topic
