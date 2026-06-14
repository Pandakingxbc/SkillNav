# VLM Memory Agent Response
# Sent from Python VLM node back to C++ MemoryAgent

Header header

# Reference to original request
int32 target_node_id           # node ID from request
uint8 request_type             # echo back request type

# === Scene Understanding Results ===
string scene_description       # VLM scene description
string inferred_room_type      # "bathroom", "bedroom", "kitchen", "living_room", "hallway", "unknown"
string[] observed_objects      # list of objects seen in the scene
float64 confidence             # 0.0-1.0 confidence in the analysis

# === Verification Results (for REQUEST_VERIFY) ===
bool target_visible            # whether target object is visible in current view
bool target_in_cluster         # whether target matches ObjectMap cluster
float64 verification_confidence  # confidence in verification result
string verification_reasoning  # VLM's reasoning for verification decision

# Verification result: three possible outcomes
uint8 VERIFY_FALSE_POSITIVE = 0   # Confirmed false positive - reduce ObjectMap confidence
uint8 VERIFY_UNCERTAIN = 1        # Uncertain - no change to confidence
uint8 VERIFY_TRUE_POSITIVE = 2    # Confirmed true positive - boost confidence
uint8 verification_result         # one of the above three values

# === Memory Agent Actions ===
bool update_memory             # whether to update node memory
bool is_false_positive         # whether this is a false positive detection (legacy, use verification_result)
float64 semantic_penalty       # penalty factor for semantic_multiplier (1.0 = no penalty, <1.0 = penalize)
float64 confidence_adjustment  # adjustment to ObjectMap confidence: <1.0 reduce, 1.0 no change, >1.0 boost

# === Additional Info ===
string reasoning               # VLM's overall reasoning
float64 processing_time        # VLM processing time in seconds
