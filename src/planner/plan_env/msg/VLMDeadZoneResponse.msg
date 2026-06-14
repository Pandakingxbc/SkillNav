# VLM Dead Zone Analysis Response
# Sent from Python VLM node back to C++ SafeAgent

Header header

# Reference to original request
int32 target_node_id           # node ID from request (-1 if none)

# VLM analysis results
string obstacle_type           # "permanent" | "temporary" | "system_boundary" | "unclear"
float64 confidence             # 0.0-1.0 confidence in the analysis
bool abandon_region            # whether to permanently abandon this region

# Recommended action
string suggested_action        # "mark_dead_zone" | "retry_later" | "reroute" | "continue_trying"
string penalty_strength        # "strong" | "moderate" | "weak" | "none"

# Additional info
string description             # VLM's description of what it sees
string reasoning               # VLM's reasoning for the decision

# Timing for retry (if temporary)
float64 retry_delay_seconds    # how long to wait before retry (0 if not applicable)
