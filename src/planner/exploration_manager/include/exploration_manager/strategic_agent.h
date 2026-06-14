#pragma once

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <memory>
#include <string>
#include <vector>

#include <plan_env/async_result_buffer.h>
#include <plan_env/StrategicLLMResponse.h>

namespace skillnav_planner {

// Forward declarations to keep this header light.
struct FSMData;
class ExplorationManager;

/**
 * Coarse navigation phase chosen by the StrategicAgent. Mirrors the four-option
 * Options-SMDP partition from the paper draft (broad → directed → approach →
 * verify). The numeric values are stable for logging / ROS messages.
 */
enum class NavigationPhase {
  BroadExploration = 0,  ///< Coverage low; favour information gain.
  DirectedSearch   = 1,  ///< Coverage rising; balance IG and SR.
  TargetApproach   = 2,  ///< A confident candidate exists; favour SR.
  Verification     = 3,  ///< Within verification distance; SR only.
};

/**
 * One strategic decision: phase + ValueMap fusion weights + (later) target
 * node selection and expected-semantics hypothesis. Returned by the LLM (or,
 * for now, by the rule-based stub) and deposited into StrategicAgent's
 * AsyncResultBuffer for the agent loop to pick up.
 *
 * The struct intentionally carries everything the consumer needs in one shot —
 * keeping the buffer single-slot (latest-wins semantics) sufficient for this
 * agent: an LLM judgment about a stale phase is worse than no judgment.
 */
struct StrategicPlan {
  NavigationPhase phase{NavigationPhase::BroadExploration};
  double w_sr{0.5};                  ///< ValueMap fusion weight, semantic relevance.
  double w_ig{0.5};                  ///< ValueMap fusion weight, info gain. (w_sr + w_ig should ≈ 1)
  int target_node_id{-1};            ///< Optional next Voronoi node hint (-1 = no override).
  std::vector<std::string> expected_semantics;  ///< Future Mnemonic hypothesis-test.
  double confidence{1.0};            ///< LLM-reported confidence; rule fallback uses 1.0.
  ros::Time issued_at;               ///< For staleness reasoning.
};

/**
 * StrategicAgent — strategic layer of the three-agent decomposition.
 *
 * Time scale
 *   ~1–5 s tick. Slow enough to afford a real LLM call (DeepSeek ~1–2 s)
 *   asynchronously. The result lives in `current_plan_` and is read by
 *   downstream consumers (ValueMap fusion, target node selection) on every
 *   fast tick at no extra cost.
 *
 * LLM connection
 *   v0 (this file): the LLM call is *stubbed* — fireLLMRequest() immediately
 *   composes a rule-based StrategicPlan (mirroring the previous
 *   ExplorationAgent fusion-weight controller) and provide()s it into the
 *   buffer. The full markFired → provide → tryConsume path is exercised even
 *   without a Python-side bridge running, so this PR can land before any
 *   prompt-engineering work.
 *
 *   v1 (follow-up): the stub is replaced by a ROS publisher to /strategic_agent
 *   /llm_request and a subscriber on /strategic_agent/llm_response, with a
 *   small Python bridge that calls DeepSeek and posts back a JSON-parsed
 *   StrategicPlan. The buffer wiring does not change.
 *
 * What this v0 does NOT do
 *   - Override ExplorationAgent's fusion weights. The rule-based controller
 *     stays the sole driver of MultiValueMapManager weights until v1 ships
 *     a verified LLM path. The plan is exposed via currentPlan() for
 *     instrumentation only — no behavioural change in this commit.
 *   - Influence target node selection. Will wire after the C1 (Hybrid
 *     Frontier-Topology) refactor lands.
 */
class StrategicAgent {
 public:
  StrategicAgent() = default;
  ~StrategicAgent() = default;

  /// One-time setup. Reads tick interval and coverage thresholds from
  /// rosparam; constructs the initial rule-based plan so currentPlan() is
  /// never invalid.
  void init(ros::NodeHandle& nh,
            std::shared_ptr<FSMData> fd,
            std::shared_ptr<ExplorationManager> em);

  /**
   * Called from the Coordinator each FSM cycle. Cheap most of the time —
   * the agent only re-fires the LLM when both:
   *   (a) `fire_interval_sec_` has elapsed since the last fire, and
   *   (b) no request is currently in flight.
   * Drains the buffer non-blocking on every call to pick up any plan that
   * landed since the previous tick.
   *
   * @return true if a freshly-computed plan was consumed into current_plan_
   *              on this tick (useful for logging / debugging).
   */
  bool tick();

  /// Latest strategic plan. Never invalid after init() — init seeds a
  /// rule-based default so consumers can always read.
  const StrategicPlan& currentPlan() const { return current_plan_; }

  /// Whether a request is currently outstanding (LLM still thinking).
  bool isThinking() const { return llm_buffer_.isInflight(); }

 private:
  /// Publish a StrategicLLMRequest on /strategic_agent/llm_request, marking
  /// the buffer as inflight. The Python bridge handles DeepSeek and posts
  /// back on /strategic_agent/llm_response.
  void fireLLMRequest();

  /// ROS callback for /strategic_agent/llm_response. Parses the message
  /// into a StrategicPlan and provide()s it into the buffer; the agent
  /// tick() then drains and applies on its own thread.
  void onLLMResponse(const plan_env::StrategicLLMResponse::ConstPtr& msg);

  /// Last-ditch rule-based plan. Only used when init() seeds current_plan_
  /// before any LLM round-trip has completed. The Python bridge also has its
  /// own (identical) rule fallback for the case where the API call fails.
  StrategicPlan computeRuleBasedFallback() const;

  // Non-owning dependencies.
  std::shared_ptr<FSMData> fd_;
  std::shared_ptr<ExplorationManager> em_;

  // Async mailbox. Single-slot is fine: only one LLM call inflight per agent.
  AsyncResultBuffer<StrategicPlan> llm_buffer_;

  // ROS I/O for the Python LLM bridge.
  ros::Publisher  llm_request_pub_;
  ros::Subscriber llm_response_sub_;

  StrategicPlan current_plan_;
  ros::Time     last_fire_time_;

  // Params (loaded from rosparam in init()).
  double fire_interval_sec_{3.0};        ///< Minimum spacing between LLM fires.
  double low_coverage_threshold_{0.3};   ///< → BroadExploration / DirectedSearch boundary.
  double high_coverage_threshold_{0.7};  ///< → DirectedSearch / TargetApproach boundary.

  /// When true, StrategicAgent is the canonical writer of MVMM fusion
  /// weights. ExplorationAgent's rule-based controller is demoted to a
  /// passive observer (still tracks room_hypothesis_, doesn't call
  /// setFusionWeights). Toggled via rosparam `~strategic_agent/
  /// enable_weight_override` (default true). For the ablation
  /// `strategic_off`, set the rosparam to false and the rule path
  /// becomes the sole driver again.
  bool enable_weight_override_{true};
};

}  // namespace skillnav_planner
