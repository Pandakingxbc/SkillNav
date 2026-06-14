#ifndef _MEMORY_AGENT_H_
#define _MEMORY_AGENT_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>

#include <plan_env/VLMMemoryRequest.h>
#include <plan_env/VLMMemoryResponse.h>

#include <plan_env/async_result_buffer.h>

namespace skillnav_planner {

class SDFMap2D;
class ObjectMap2D;
class VoronoiTopology;
class FrontierMap2D;
struct ObjectCluster;
struct TopologyNode;

/**
 * Result of one asynchronous VLM verification on a candidate target.
 * Populated by MemoryAgent::vlmResponseCallback from a parsed VLMMemoryResponse
 * and deposited into the agent's AsyncResultQueue. The agent loop later drains
 * the queue and applies the result (votes, frame counter, topology memory)
 * inside applyPendingVerifications().
 *
 * Keeping the ROS message type out of the queued payload lets the
 * (potentially testable) state-mutation code be exercised without a ROS master.
 */
struct CandidateVerificationResult {
  int target_node_id{-1};               ///< Anchor node id used for routing
                                        ///< back to the right candidate.
  uint8_t verification_result{0};       ///< VLMMemoryResponse::VERIFY_* enum value.
  bool update_memory{false};            ///< Whether the response carries a
                                        ///< scene description to attach to the node.
  std::string scene_description;
  std::string inferred_room_type;
  std::vector<std::string> observed_objects;
  double confidence{0.0};
};

/**
 * @brief Candidate target for object navigation
 *
 * Unifies information from ObjectMap2D, ValueMap, and VoronoiTopology.
 * Used by Memory Agent for opportunity cost calculation and FP tracking.
 */
struct CandidateTarget {
  // === From ObjectMap2D ===
  int cluster_id;                    ///< ObjectMap cluster ID
  Eigen::Vector2d position;          ///< Centroid position
  int best_label;                    ///< Semantic class label
  double detection_confidence;       ///< Detector confidence [0,1]
  int observation_count;             ///< Number of observations

  // === From ValueMap ===
  double semantic_value;             ///< ITM score at position
  double value_confidence;           ///< Confidence of semantic value

  // === From VoronoiTopology ===
  int anchor_node_id;                ///< Nearest Voronoi node ID
  double distance_to_node;           ///< Distance to anchor node

  // === False Positive Tracking ===
  int verification_count;            ///< Times verified close-up
  int false_positive_count;          ///< Times rejected as FP
  double cumulative_confidence;      ///< Historical confidence (decay)
  ros::Time last_observation;        ///< Last observation time
  ros::Time last_verification;       ///< Last VLM verification time
  bool vlm_pending;                  ///< Whether VLM request is in flight

  // === Single-Shot Verification State (2026-05-19 refactor) ===
  // The Python verifier (vlm_memory_agent_node.py) sends N annotated RGB frames
  // in ONE Qwen-VL call with an InfoNav-style prompt and returns a 3-level
  // decision. No multi-distance staircase, no majority voting.
  // Verification fires once per candidate, when the robot crosses
  // VERIFY_TRIGGER_DIST during approach.
  static constexpr double VERIFY_TRIGGER_DIST = 1.0;  ///< meters from candidate

  bool verification_complete;        ///< Whether the single VLM call returned
  uint8_t final_verification_result; ///< Result returned by the VLM call

  // === Computed Metrics ===
  double travel_cost;                ///< Path cost from robot
  double opportunity_cost;           ///< Final ranking metric (lower = better)

  CandidateTarget()
    : cluster_id(-1), best_label(-1), detection_confidence(0.0),
      observation_count(0), semantic_value(0.0), value_confidence(0.0),
      anchor_node_id(-1), distance_to_node(0.0),
      verification_count(0), false_positive_count(0),
      cumulative_confidence(0.0), vlm_pending(false),
      verification_complete(false),
      final_verification_result(1),  // Default UNCERTAIN
      travel_cost(0.0), opportunity_cost(std::numeric_limits<double>::max()) {}

  /**
   * @brief Reset verification state for new approach
   */
  void resetVerificationState() {
    verification_complete = false;
    final_verification_result = 1;  // UNCERTAIN
    vlm_pending = false;
  }

  /**
   * @brief Single-shot trigger: fire once when robot is inside trigger radius.
   */
  bool shouldVerifyAtDistance(double current_dist,
                              double trigger_dist = VERIFY_TRIGGER_DIST) const {
    if (verification_complete) return false;
    if (vlm_pending) return false;
    return current_dist <= trigger_dist;
  }

  /**
   * @brief Compute reliability score combining all sources
   *
   * Higher = more reliable target
   */
  double getReliabilityScore() const {
    // Base: detection confidence * semantic value
    double base = detection_confidence * std::max(0.1, semantic_value);

    // Observation bonus (saturates at 5 observations)
    double obs_bonus = std::min(1.0, observation_count / 5.0);

    // False positive penalty (exponential decay)
    double fp_penalty = std::exp(-0.5 * false_positive_count);

    // Time decay (older observations less reliable)
    double time_since_obs = (ros::Time::now() - last_observation).toSec();
    double time_decay = std::exp(-0.01 * time_since_obs);  // ~63% at 100s

    return base * (0.5 + 0.5 * obs_bonus) * fp_penalty * time_decay;
  }

  /**
   * @brief Check if this target needs VLM verification
   */
  bool needsVerification(double verify_distance) const {
    // Need verification if:
    // 1. Has enough observations to be worth verifying
    // 2. Hasn't been verified recently
    // 3. No VLM request pending
    // 4. Robot is close enough
    if (observation_count < 2) return false;
    if (vlm_pending) return false;
    if (verification_count > 0) {
      // Already verified, only re-verify if FP detected
      return false;
    }
    return true;
  }
};

/**
 * @brief Memory Agent for SkillNav Multi-Agent System
 *
 * Responsibilities (from architecture doc):
 * - Candidate target management (sync with ObjectMap2D)
 * - False positive tracking (verification via VLM)
 * - Opportunity cost calculation
 * - Voronoi-anchored spatial indexing
 *
 * Design: Rule-based (~100ms) + VLM-assisted (async ~2s)
 * - Core logic is rule-based for fast response
 * - VLM is called asynchronously for verification
 */
class MemoryAgent {
public:
  MemoryAgent() = default;
  ~MemoryAgent() = default;

  /**
   * @brief Initialize the memory agent
   */
  void init(ros::NodeHandle& nh,
            const std::shared_ptr<SDFMap2D>& sdf_map,
            ObjectMap2D* object_map,
            VoronoiTopology* topology,
            FrontierMap2D* frontier_map);

  /**
   * @brief Update candidate targets from ObjectMap2D
   *
   * Call this periodically (~10Hz) to sync with object detections.
   * Fast rule-based update, no VLM calls.
   */
  void updateCandidates();

  /**
   * @brief Set current target object for navigation
   */
  void setTargetObject(const std::string& target) { target_object_ = target; }

  /**
   * @brief Check if any candidates need VLM verification
   *
   * Call this when robot is close to a candidate.
   * Will send async VLM request if needed.
   *
   * @param robot_pos Current robot position
   * @param robot_yaw Current robot heading
   * @return True if VLM request was sent
   */
  bool checkVerification(const Eigen::Vector2d& robot_pos, double robot_yaw);

  /**
   * @brief Get ranked candidate targets (lower opportunity_cost = better)
   */
  std::vector<CandidateTarget> getRankedTargets() const;

  /**
   * @brief Get best candidate target
   * @return Pointer to best target, or nullptr if none
   */
  const CandidateTarget* getBestTarget() const;

  /**
   * @brief Get nearby memories as context string for VLM
   */
  std::string getNearbyMemoriesString(const Eigen::Vector2d& pos) const;

  /**
   * @brief Record successful verification (target found)
   */
  void recordSuccess(int cluster_id);

  /**
   * @brief Record false positive (target not found after navigation)
   */
  void recordFalsePositive(int cluster_id);

  /**
   * @brief Get the current navigation target cluster ID
   * Returns -1 if no target is being navigated to
   */
  int getCurrentNavigationTarget() const { return current_nav_target_; }

  /**
   * @brief Set the current navigation target (called when robot starts navigating to object)
   */
  void setCurrentNavigationTarget(int cluster_id);

  /**
   * @brief Clear current navigation target
   */
  void clearNavigationTarget() { current_nav_target_ = -1; }

  // Accessors
  const std::vector<CandidateTarget>& getCandidates() const { return candidates_; }
  int getVLMCallsThisMinute() const { return vlm_call_timestamps_.size(); }

  /**
   * @brief Lookup verification state for the candidate nearest to `pos`
   *        (within `match_radius`). Used by the FSM as a synchronous gate
   *        before declaring REACH_OBJECT.
   *
   * @param[out] pending          Whether a VLM call is in flight
   * @param[out] complete         Whether a final decision has been reached
   * @param[out] is_false_positive Whether final decision is FALSE_POSITIVE
   * @return true if a candidate was found near `pos`, false otherwise.
   */
  bool queryVerificationNear(const Eigen::Vector2d& pos,
                             double match_radius,
                             bool& pending,
                             bool& complete,
                             bool& is_false_positive) const;

private:
  // VLM communication
  void vlmResponseCallback(const plan_env::VLMMemoryResponse::ConstPtr& msg);
  void sendVLMRequest(int target_idx, uint8_t request_type,
                      const Eigen::Vector2d& robot_pos, double robot_yaw);
  bool checkThrottle();
  void recordVLMCall();

  // Candidate management
  void syncWithObjectMap();
  void linkToTopologyNodes();
  void computeOpportunityCosts(const Eigen::Vector2d& robot_pos);

  /**
   * @brief Compute opportunity cost for a candidate target
   *
   * OpportunityCost = (TravelCost + ForegoneExploration + FPRisk) / Reliability
   */
  double computeOpportunityCost(const CandidateTarget& target,
                                const Eigen::Vector2d& robot_pos) const;

  /**
   * @brief Apply single-shot VLM verification result to ObjectMap and topology
   */
  void applyFinalVerificationResult(CandidateTarget& candidate);

  /**
   * @brief Drain vlm_queue_ and apply each pending verification result to its
   *        candidate. Called from updateCandidates() at the top of each tick,
   *        so all candidate-state mutations happen on the agent loop's thread
   *        (the ROS callback only enqueues).
   */
  void applyPendingVerifications();

  // Data
  std::vector<CandidateTarget> candidates_;
  std::unordered_map<int, int> cluster_to_candidate_;  ///< ObjectMap cluster_id -> candidate index

  // Component references
  std::shared_ptr<SDFMap2D> sdf_map_;
  ObjectMap2D* object_map_;
  VoronoiTopology* topology_;
  FrontierMap2D* frontier_map_;

  // Navigation context
  std::string target_object_;
  Eigen::Vector2d last_robot_pos_;
  int current_nav_target_ = -1;  ///< Cluster ID of current navigation target

  // VLM communication
  ros::Publisher vlm_request_pub_;
  ros::Subscriber vlm_response_sub_;
  std::deque<double> vlm_call_timestamps_;  ///< Timestamps of recent VLM calls

  /// Multi-slot fan-in mailbox. ROS callback parses each response and pushes
  /// here; updateCandidates() drains and routes by anchor_node_id. Used in
  /// place of the previous direct candidate mutation inside the callback so
  /// the entire state machine lives on a single thread.
  AsyncResultQueue<CandidateVerificationResult> vlm_queue_;

  // Parameters
  int max_vlm_calls_per_minute_;     ///< VLM throttling
  double verify_distance_;           ///< Distance to trigger verification
  double memory_search_radius_;      ///< Radius for nearby memory search
  double fp_penalty_weight_;         ///< Weight for FP risk in opportunity cost
  double foregone_exploration_weight_; ///< Weight for foregone exploration

  // Reward/penalty policy
  double semantic_boost_factor_;     ///< Per-success multiplicative boost for semantic_multiplier
  double semantic_boost_cap_;        ///< Upper cap on semantic_multiplier after boost
  double confidence_success_boost_;  ///< Per-success multiplicative boost for cumulative_confidence
  double confidence_fp_penalty_;     ///< Per-FP multiplicative penalty for cumulative_confidence
  double cluster_success_boost_;     ///< Boost passed to ObjectMap::adjustClusterConfidence on TP
  double cluster_fp_penalty_;        ///< Penalty passed to ObjectMap::adjustClusterConfidence on FP
  double node_fp_penalty_strong_;    ///< Node semantic_multiplier factor on multi-frame FP
  double node_fp_penalty_weak_;      ///< Node semantic_multiplier factor on single recordFalsePositive

  // State
  bool initialized_ = false;
};

}  // namespace skillnav_planner

#endif  // _MEMORY_AGENT_H_
