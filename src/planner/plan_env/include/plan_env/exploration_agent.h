#ifndef _EXPLORATION_AGENT_H_
#define _EXPLORATION_AGENT_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <string>

namespace skillnav_planner {

class MultiValueMapManager;
class VoronoiTopology;
class MemoryAgent;

/**
 * @brief Exploration Agent — intermittent SR/IG fusion-weight controller.
 *
 * Design (per SkillNav multi-agent architecture):
 *   - Long-running rule layer (this skeleton): coverage-tier mapping to weights
 *   - Intermittent LLM layer (TODO): consults Voronoi memory + room hypothesis
 *     to rewrite fusion weights and IG prompt text
 *
 * This skeleton intentionally avoids LLM calls; it exposes the activation
 * surface so the planner can drive ExplorationAgent every loop without cost.
 * Real LLM-driven decisions plug into tick() via a future callLLMForDecision().
 */
class ExplorationAgent {
public:
  ExplorationAgent() = default;
  ~ExplorationAgent() = default;

  void init(ros::NodeHandle& nh,
            MultiValueMapManager* mvm,
            VoronoiTopology* topology,
            MemoryAgent* memory);

  /**
   * @brief Periodic tick. Cheap; safe to call every planning cycle.
   * @param robot_pos current robot position
   * @param exploration_pct fraction of map explored [0, 1]
   */
  void tick(const Eigen::Vector2d& robot_pos, double exploration_pct);

  /** @brief Current room-type hypothesis (for IG prompt selection in future LLM step). */
  std::string getRoomHypothesis() const { return room_hypothesis_; }

  /** @brief Force a manual weight override (debugging / ablation). Auto-normalized. */
  void setManualWeights(double w_sr, double w_ig);

  /** @brief Disable manual override and resume rule-based scheduling. */
  void clearManualOverride() { manual_override_ = false; }

  bool isManualOverride() const { return manual_override_; }

  /**
   * @brief Demote the rule-based weight controller to a no-op writer.
   *
   * When StrategicAgent (a higher-time-scale agent) is configured to own
   * the ValueMap fusion weights, this method is called once during init so
   * that the per-cycle tick() still computes the room-type hypothesis (used
   * for prompt selection elsewhere) but does not call setFusionWeights().
   * The intent is "rule-based controller becomes a passive observer."
   *
   * Off by default — when StrategicAgent's enable_weight_override flag is
   * false the rule path stays the sole writer (ablation: `strategic_off`).
   */
  void setWeightsExternallyOwned(bool externally_owned) {
    weights_externally_owned_ = externally_owned;
  }

  bool isWeightsExternallyOwned() const { return weights_externally_owned_; }

private:
  // Pick (w_sr, w_ig) from exploration progress.
  // Early: favor IG (broad coverage). Late: favor SR (target-direct).
  void applyCoverageTierWeights(double exploration_pct);

  // Component references (not owned)
  MultiValueMapManager* mvm_ = nullptr;
  VoronoiTopology* topology_ = nullptr;
  MemoryAgent* memory_ = nullptr;

  // State
  std::string room_hypothesis_ = "broad_exploration";  // default
  bool manual_override_ = false;
  /// When true, the StrategicAgent owns fusion weights and this controller
  /// stops calling setFusionWeights. Set via setWeightsExternallyOwned(true)
  /// during StrategicAgent::init when override is enabled.
  bool weights_externally_owned_ = false;
  double last_w_sr_ = 0.5;
  double last_w_ig_ = 0.5;
  double last_exploration_pct_ = -1.0;
  ros::Time last_log_time_;

  // Parameters
  double low_coverage_threshold_ = 0.30;
  double high_coverage_threshold_ = 0.70;
  double weight_change_eps_ = 0.05;  // hysteresis to avoid log spam

  // Tier fusion weights (defaults preserve the prior hard-coded policy)
  double tier_broad_w_sr_    = 0.3;
  double tier_broad_w_ig_    = 0.7;
  double tier_directed_w_sr_ = 0.5;
  double tier_directed_w_ig_ = 0.5;
  double tier_commit_w_sr_   = 0.8;
  double tier_commit_w_ig_   = 0.2;
};

}  // namespace skillnav_planner

#endif  // _EXPLORATION_AGENT_H_
