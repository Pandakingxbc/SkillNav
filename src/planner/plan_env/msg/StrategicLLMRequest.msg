# Strategic LLM request — sent from C++ StrategicAgent to the Python bridge
# that calls DeepSeek. Kept minimal in v1; later fields can be appended (msg
# fields are append-safe in ROS).

Header header

string target_object              # e.g. "toilet", from /target_object rosparam
float64 coverage_estimate         # 0.0–1.0, fraction of map explored

int32 current_phase               # NavigationPhase enum {0..3}
int32 candidate_count             # number of detected target-class candidates
int32 fp_count                    # historical false-positive count
int32 stuck_count                 # historical stuck-recovery events
