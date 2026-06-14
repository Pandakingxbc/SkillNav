#include <plan_env/exploration_agent.h>
#include <plan_env/multi_valuemap_manager.h>
#include <plan_env/voronoi_topology.h>
#include <plan_env/memory_agent.h>

#include <cmath>

namespace skillnav_planner {

void ExplorationAgent::init(ros::NodeHandle& nh,
                            MultiValueMapManager* mvm,
                            VoronoiTopology* topology,
                            MemoryAgent* memory) {
  mvm_ = mvm;
  topology_ = topology;
  memory_ = memory;

  nh.param("exploration_agent/low_coverage_threshold",
           low_coverage_threshold_, 0.30);
  nh.param("exploration_agent/high_coverage_threshold",
           high_coverage_threshold_, 0.70);

  // Tier fusion weights — overridable, defaults match prior hard-coded policy.
  nh.param("exploration_agent/tier_broad_w_sr",    tier_broad_w_sr_,    0.3);
  nh.param("exploration_agent/tier_broad_w_ig",    tier_broad_w_ig_,    0.7);
  nh.param("exploration_agent/tier_directed_w_sr", tier_directed_w_sr_, 0.5);
  nh.param("exploration_agent/tier_directed_w_ig", tier_directed_w_ig_, 0.5);
  nh.param("exploration_agent/tier_commit_w_sr",   tier_commit_w_sr_,   0.8);
  nh.param("exploration_agent/tier_commit_w_ig",   tier_commit_w_ig_,   0.2);

  last_log_time_ = ros::Time::now();
  ROS_INFO("[ExplorationAgent] Initialized (rule-based, LLM placeholder). "
           "Tiers: low<%.2f, high>%.2f", low_coverage_threshold_, high_coverage_threshold_);
}

void ExplorationAgent::tick(const Eigen::Vector2d& /*robot_pos*/, double exploration_pct) {
  if (!mvm_) return;
  if (manual_override_) return;
  applyCoverageTierWeights(exploration_pct);
}

void ExplorationAgent::applyCoverageTierWeights(double exploration_pct) {
  // Tier mapping:
  //   pct <  low        : early — favor IG (broad coverage, find rooms)
  //   low <= pct < high : middle — balanced
  //   pct >= high       : late — favor SR (commit to target-direct evidence)
  double w_sr, w_ig;
  if (exploration_pct < low_coverage_threshold_) {
    w_sr = tier_broad_w_sr_; w_ig = tier_broad_w_ig_;
    room_hypothesis_ = "broad_exploration";
  } else if (exploration_pct < high_coverage_threshold_) {
    w_sr = tier_directed_w_sr_; w_ig = tier_directed_w_ig_;
    room_hypothesis_ = "directed_search";
  } else {
    w_sr = tier_commit_w_sr_; w_ig = tier_commit_w_ig_;
    room_hypothesis_ = "target_commit";
  }

  bool changed = std::fabs(w_sr - last_w_sr_) > weight_change_eps_ ||
                 std::fabs(w_ig - last_w_ig_) > weight_change_eps_;
  if (changed) {
    // Skip the setFusionWeights call when StrategicAgent is the canonical
    // weight writer (its plan is consumed every ~3 s and would otherwise
    // overwrite us back-and-forth). Still track room_hypothesis_ +
    // last_w_* because other code (and a future LLM prompt selector)
    // reads them.
    if (!weights_externally_owned_) {
      mvm_->setFusionWeights(w_sr, w_ig);
      ROS_INFO("[ExplorationAgent] Tier shift: pct=%.2f hypothesis=%s weights(SR=%.2f, IG=%.2f)",
               exploration_pct, room_hypothesis_.c_str(), w_sr, w_ig);
    } else {
      ROS_DEBUG("[ExplorationAgent] Tier shift observed (Strategic owns weights): "
                "pct=%.2f hypothesis=%s would-be-weights(SR=%.2f, IG=%.2f)",
                exploration_pct, room_hypothesis_.c_str(), w_sr, w_ig);
    }
    last_w_sr_ = w_sr;
    last_w_ig_ = w_ig;
  }
  last_exploration_pct_ = exploration_pct;
}

void ExplorationAgent::setManualWeights(double w_sr, double w_ig) {
  manual_override_ = true;
  if (mvm_) mvm_->setFusionWeights(w_sr, w_ig);
  last_w_sr_ = w_sr;
  last_w_ig_ = w_ig;
  ROS_WARN("[ExplorationAgent] MANUAL override active: SR=%.3f IG=%.3f", w_sr, w_ig);
}

}  // namespace skillnav_planner
