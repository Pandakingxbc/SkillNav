#include <exploration_manager/strategic_agent.h>

#include <exploration_manager/exploration_data.h>
#include <exploration_manager/exploration_manager.h>
#include <plan_env/exploration_agent.h>          // demote rule controller
#include <plan_env/multi_valuemap_manager.h>     // setFusionWeights
#include <plan_env/sdf_map2d.h>                  // getExploredFraction
#include <plan_env/memory_agent.h>               // candidate / FP context
#include <plan_env/StrategicLLMRequest.h>        // outgoing LLM request

namespace skillnav_planner {

void StrategicAgent::init(ros::NodeHandle& nh,
                          std::shared_ptr<FSMData> fd,
                          std::shared_ptr<ExplorationManager> em)
{
  fd_ = fd;
  em_ = em;

  nh.param("strategic_agent/fire_interval_sec",   fire_interval_sec_,    3.0);
  nh.param("strategic_agent/low_coverage_thresh", low_coverage_threshold_,  0.3);
  nh.param("strategic_agent/high_coverage_thresh",high_coverage_threshold_, 0.7);
  nh.param("strategic_agent/enable_weight_override", enable_weight_override_, true);

  // Seed with a rule-based plan so currentPlan() is always valid.
  current_plan_ = computeRuleBasedFallback();
  current_plan_.issued_at = ros::Time::now();
  last_fire_time_ = ros::Time(0);  // force first fire to happen on first tick

  // If override is enabled, demote the legacy rule-based controller from
  // writing MVMM weights so our writes don't ping-pong with its writes.
  // ExplorationAgent::tick still runs each cycle to maintain
  // room_hypothesis_; it just stops calling setFusionWeights.
  if (enable_weight_override_ && em_ && em_->exploration_agent_) {
    em_->exploration_agent_->setWeightsExternallyOwned(true);
  }

  // LLM bridge wiring. The Python bridge (strategic_agent_node.py) translates
  // requests into DeepSeek calls and posts JSON-parsed responses back.
  llm_request_pub_ = nh.advertise<plan_env::StrategicLLMRequest>(
      "/strategic_agent/llm_request", 10);
  llm_response_sub_ = nh.subscribe(
      "/strategic_agent/llm_response", 10, &StrategicAgent::onLLMResponse, this);

  ROS_INFO("[StrategicAgent] initialized: fire_interval=%.1fs, "
           "thresholds=[%.2f, %.2f], weight_override=%s",
           fire_interval_sec_, low_coverage_threshold_, high_coverage_threshold_,
           enable_weight_override_ ? "ON (Strategic writes MVMM)"
                                   : "OFF (ExplorationAgent writes MVMM)");
}

bool StrategicAgent::tick()
{
  bool plan_updated = false;

  // 1. Drain any LLM result deposited by the (currently stubbed) async path.
  StrategicPlan landed;
  if (llm_buffer_.tryConsume(landed)) {
    current_plan_ = landed;
    plan_updated = true;
    ROS_INFO("[StrategicAgent] new plan: phase=%d, w_sr=%.2f, w_ig=%.2f, conf=%.2f",
             static_cast<int>(current_plan_.phase),
             current_plan_.w_sr, current_plan_.w_ig, current_plan_.confidence);

    // Push the new fusion weights down to the MultiValueMapManager so the
    // next cycle's frontier scoring uses them. Only when override is on;
    // otherwise leave MVMM under the ExplorationAgent rule controller.
    if (enable_weight_override_ && em_ && em_->multi_valuemap_manager_) {
      em_->multi_valuemap_manager_->setFusionWeights(current_plan_.w_sr,
                                                     current_plan_.w_ig);
    }
  }

  // 2. Fire a fresh request if it's time and nothing is outstanding.
  const ros::Time now = ros::Time::now();
  const bool interval_elapsed = (now - last_fire_time_).toSec() >= fire_interval_sec_;
  if (interval_elapsed && !llm_buffer_.isInflight()) {
    fireLLMRequest();
    last_fire_time_ = now;
  }

  return plan_updated;
}

void StrategicAgent::fireLLMRequest()
{
  llm_buffer_.markFired();

  plan_env::StrategicLLMRequest req;
  req.header.stamp = ros::Time::now();
  req.header.frame_id = "world";

  // /target_object is set by Habitat at episode start (habitat_evaluation.py).
  // Fall back to empty string if not yet present — bridge tolerates "unknown".
  std::string target_object;
  ros::param::get("/target_object", target_object);
  req.target_object = target_object;

  // Coverage: walk the SDFMap2D occupancy buffer for the fraction of cells
  // we've ever observed. O(N) but cheap at 3 s cadence.
  req.coverage_estimate = (em_ && em_->sdf_map_)
                              ? em_->sdf_map_->getExploredFraction()
                              : 0.0;

  // Candidate + FP context: aggregated from MemoryAgent. Use it via the
  // (still EM-owned) raw pointer rather than the Coordinator's view so we
  // don't need to plumb that handle in this commit.
  int candidate_count = 0;
  int fp_count = 0;
  if (em_ && em_->memory_agent_) {
    const auto& cands = em_->memory_agent_->getCandidates();
    candidate_count = static_cast<int>(cands.size());
    for (const auto& c : cands) fp_count += c.false_positive_count;
  }
  req.candidate_count = candidate_count;
  req.fp_count        = fp_count;

  // Stuck recovery history: SafeAgent appends here on each TRYOUT_FAILED.
  req.stuck_count = fd_ ? static_cast<int>(fd_->stucking_points_.size()) : 0;

  req.current_phase = static_cast<int>(current_plan_.phase);

  llm_request_pub_.publish(req);
}

void StrategicAgent::onLLMResponse(
    const plan_env::StrategicLLMResponse::ConstPtr& msg)
{
  // ROS callback thread — parse only, no side effects on em_/MVMM here.
  // The agent loop drains the buffer next tick() and applies the plan.
  if (!llm_buffer_.isInflight()) {
    ROS_WARN("[StrategicAgent] got LLM response with no request inflight — "
             "discarding (stale).");
    return;
  }

  StrategicPlan plan;
  // Clamp phase to enum range; the bridge sanitises but be defensive.
  int phase_raw = msg->phase;
  if (phase_raw < 0) phase_raw = 0;
  if (phase_raw > 3) phase_raw = 3;
  plan.phase = static_cast<NavigationPhase>(phase_raw);
  plan.w_sr           = msg->w_sr;
  plan.w_ig           = msg->w_ig;
  plan.target_node_id = msg->target_node_id;
  plan.expected_semantics.assign(msg->expected_semantics.begin(),
                                 msg->expected_semantics.end());
  plan.confidence     = msg->confidence;
  plan.issued_at      = ros::Time::now();

  llm_buffer_.provide(plan);

  ROS_INFO("[StrategicAgent] LLM reply queued: phase=%d w=(%.2f,%.2f) "
           "conf=%.2f tok=(%d,%d) latency=%.2fs",
           static_cast<int>(plan.phase), plan.w_sr, plan.w_ig,
           plan.confidence, msg->prompt_tokens, msg->completion_tokens,
           msg->latency_seconds);
}

StrategicPlan StrategicAgent::computeRuleBasedFallback() const
{
  StrategicPlan p;
  p.confidence = 1.0;  // rule version is deterministic

  // Coverage proxy. The real estimate should come from FrontierMap2D; for
  // now mirror what ExplorationAgent::tick() does (a placeholder 0.5).
  // TODO: hook FrontierMap2D::getExploredFraction() once it exists.
  const double coverage = 0.5;

  if (coverage < low_coverage_threshold_) {
    p.phase = NavigationPhase::BroadExploration;
    p.w_sr = 0.3;
    p.w_ig = 0.7;
  } else if (coverage > high_coverage_threshold_) {
    p.phase = NavigationPhase::TargetApproach;
    p.w_sr = 0.8;
    p.w_ig = 0.2;
  } else {
    p.phase = NavigationPhase::DirectedSearch;
    p.w_sr = 0.5;
    p.w_ig = 0.5;
  }

  return p;
}

}  // namespace skillnav_planner
