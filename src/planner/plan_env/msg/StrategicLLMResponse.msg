# Strategic LLM response — emitted by the Python bridge after one DeepSeek
# call. Mirrors StrategicPlan in strategic_agent.h.

Header header

int32 phase                       # 0=BROAD_EXPLORATION, 1=DIRECTED_SEARCH,
                                  # 2=TARGET_APPROACH, 3=VERIFICATION
float64 w_sr                      # ValueMap fusion weight: semantic relevance
float64 w_ig                      # ValueMap fusion weight: information gain
                                  # (w_sr + w_ig should ≈ 1.0)

int32 target_node_id              # optional Voronoi node id; -1 if not specified

string[] expected_semantics       # objects expected en route — fuels future
                                  # Mnemonic hypothesis-test loop

float64 confidence                # LLM-reported confidence 0.0–1.0
string reasoning                  # one-line explanation for logging (truncated)

float64 latency_seconds           # observed LLM call latency, set by bridge
int32 prompt_tokens               # token accounting for paper economy table
int32 completion_tokens
