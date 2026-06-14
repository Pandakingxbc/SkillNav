#include <plan_env/memory_agent.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/voronoi_topology.h>
#include <plan_env/frontier_map2d.h>

#include <algorithm>
#include <unordered_set>

namespace skillnav_planner {

void MemoryAgent::init(ros::NodeHandle& nh,
                       const std::shared_ptr<SDFMap2D>& sdf_map,
                       ObjectMap2D* object_map,
                       VoronoiTopology* topology,
                       FrontierMap2D* frontier_map) {
  sdf_map_ = sdf_map;
  object_map_ = object_map;
  topology_ = topology;
  frontier_map_ = frontier_map;

  // Load parameters
  nh.param("memory_agent/max_vlm_calls_per_minute", max_vlm_calls_per_minute_, 10);
  // Trigger early (2.0m) so Qwen-VL (~7s latency) usually returns by the time
  // the agent reaches REACH_DISTANCE (0.2m). Hover override removed
  // 2026-05-21 — agent walks naturally; the FSM gate at reach distance
  // still blocks REACH on pending/rejected as the safety net.
  nh.param("memory_agent/verify_distance", verify_distance_, 2.0);
  nh.param("memory_agent/memory_search_radius", memory_search_radius_, 2.0);
  nh.param("memory_agent/fp_penalty_weight", fp_penalty_weight_, 1.0);
  nh.param("memory_agent/foregone_exploration_weight", foregone_exploration_weight_, 0.3);

  // Pick up the navigation target from rosparam (habitat_evaluation.py sets
  // /target_object at episode start). Without this, target_object_ stays empty
  // and the Python verifier falls back to UNCERTAIN. Polling-with-fallback so
  // we never block init even if Habitat is late.
  {
    std::string target;
    double wait_sec = 5.0;
    ros::Time deadline = ros::Time::now() + ros::Duration(wait_sec);
    while (ros::ok() && ros::Time::now() < deadline) {
      if (nh.getParam("/target_object", target) && !target.empty()) break;
      ros::Duration(0.2).sleep();
    }
    if (!target.empty()) {
      target_object_ = target;
      ROS_INFO("[MemoryAgent] target_object='%s' (from /target_object)", target.c_str());
    } else {
      ROS_WARN("[MemoryAgent] /target_object not set within %.1fs; target_object_ left empty",
               wait_sec);
    }
  }

  // Reward/penalty policy (literals previously inlined at TP/FP handlers)
  nh.param("memory_agent/semantic_boost_factor",    semantic_boost_factor_,    1.1);
  nh.param("memory_agent/semantic_boost_cap",       semantic_boost_cap_,       1.5);
  nh.param("memory_agent/confidence_success_boost", confidence_success_boost_, 1.2);
  nh.param("memory_agent/confidence_fp_penalty",    confidence_fp_penalty_,    0.5);
  nh.param("memory_agent/cluster_success_boost",    cluster_success_boost_,    1.3);
  nh.param("memory_agent/cluster_fp_penalty",       cluster_fp_penalty_,       0.5);
  nh.param("memory_agent/node_fp_penalty_strong",   node_fp_penalty_strong_,   0.5);
  nh.param("memory_agent/node_fp_penalty_weak",     node_fp_penalty_weak_,     0.7);

  // ROS communication
  vlm_request_pub_ = nh.advertise<plan_env::VLMMemoryRequest>(
      "/memory_agent/vlm_request", 10);
  vlm_response_sub_ = nh.subscribe("/memory_agent/vlm_response", 10,
      &MemoryAgent::vlmResponseCallback, this);

  initialized_ = true;

  ROS_INFO("[MemoryAgent] Initialized with verify_dist=%.2f, max_vlm_calls=%d/min",
           verify_distance_, max_vlm_calls_per_minute_);
}

void MemoryAgent::updateCandidates() {
  if (!initialized_ || !object_map_ || !topology_) return;

  // Step 0: Drain pending VLM verification responses (deposited by the ROS
  // callback) and apply each to its candidate. Doing this first ensures
  // subsequent steps see the freshest verification state.
  applyPendingVerifications();

  // Step 1: Sync with ObjectMap2D
  syncWithObjectMap();

  // Step 2: Link to topology nodes
  linkToTopologyNodes();

  // Step 3: Compute opportunity costs
  computeOpportunityCosts(last_robot_pos_);
}

void MemoryAgent::syncWithObjectMap() {
  // Get objects from ObjectMap2D
  std::vector<std::vector<Eigen::Vector2d>> clusters;
  std::vector<Eigen::Vector2d> averages;
  std::vector<int> labels;
  object_map_->getObjects(clusters, averages, labels);

  // Track which cluster IDs we've seen
  std::unordered_set<int> seen_clusters;

  for (size_t i = 0; i < averages.size(); ++i) {
    int cluster_id = static_cast<int>(i);  // Use index as cluster ID
    seen_clusters.insert(cluster_id);

    auto it = cluster_to_candidate_.find(cluster_id);
    if (it == cluster_to_candidate_.end()) {
      // New cluster - create candidate
      CandidateTarget candidate;
      candidate.cluster_id = cluster_id;
      candidate.position = averages[i];
      candidate.best_label = labels[i];
      candidate.observation_count = 1;
      candidate.last_observation = ros::Time::now();
      candidate.cumulative_confidence = 0.5;  // Initial confidence

      candidates_.push_back(candidate);
      cluster_to_candidate_[cluster_id] = candidates_.size() - 1;
    } else {
      // Existing cluster - update
      int idx = it->second;
      candidates_[idx].position = averages[i];
      candidates_[idx].best_label = labels[i];
      candidates_[idx].observation_count++;
      candidates_[idx].last_observation = ros::Time::now();
      // Decay and add new confidence
      candidates_[idx].cumulative_confidence =
          0.9 * candidates_[idx].cumulative_confidence + 0.1;
    }
  }

  // Remove candidates for clusters that no longer exist
  // (Not removing for now - keep memory of past detections)
}

void MemoryAgent::linkToTopologyNodes() {
  if (!topology_) return;

  for (auto& candidate : candidates_) {
    TopologyNode* nearest = topology_->getNearestNode(candidate.position);
    if (nearest) {
      candidate.anchor_node_id = nearest->id;
      candidate.distance_to_node = (candidate.position - nearest->position).norm();

      // Get semantic value from node
      candidate.semantic_value = nearest->getEffectiveValue();
    } else {
      candidate.anchor_node_id = -1;
      candidate.distance_to_node = -1;
    }
  }
}

void MemoryAgent::computeOpportunityCosts(const Eigen::Vector2d& robot_pos) {
  for (auto& candidate : candidates_) {
    // Compute travel cost
    candidate.travel_cost = (robot_pos - candidate.position).norm();
    // Compute opportunity cost
    candidate.opportunity_cost = computeOpportunityCost(candidate, robot_pos);
  }
}

double MemoryAgent::computeOpportunityCost(const CandidateTarget& target,
                                           const Eigen::Vector2d& robot_pos) const {
  // 1. Direct cost: Euclidean distance (simple approximation)
  double path_cost = (robot_pos - target.position).norm();

  // 2. Foregone exploration: value of unexplored areas we'd skip
  double foregone_exploration = 0.0;
  if (topology_ && frontier_map_) {
    const auto& nodes = topology_->getNodes();
    for (const auto& node : nodes) {
      if (node.type == TopologyNode::FRONTIER_ADJACENT) {
        double value = node.getEffectiveValue();
        double accessibility = 1.0 / (1.0 + (node.position - robot_pos).norm());
        foregone_exploration += value * accessibility;
      }
    }
  }

  // 3. False positive risk
  double fp_risk = target.false_positive_count * fp_penalty_weight_;

  // 4. Reliability score (higher = better)
  double reliability = target.getReliabilityScore();

  // Combined opportunity cost (lower is better)
  double cost = (path_cost + foregone_exploration * foregone_exploration_weight_ + fp_risk)
                / (reliability + 0.01);  // Add small epsilon to avoid division by zero

  return cost;
}

std::vector<CandidateTarget> MemoryAgent::getRankedTargets() const {
  std::vector<CandidateTarget> ranked = candidates_;

  // Sort by opportunity cost (lower = better)
  std::sort(ranked.begin(), ranked.end(),
      [](const CandidateTarget& a, const CandidateTarget& b) {
        return a.opportunity_cost < b.opportunity_cost;
      });

  return ranked;
}

const CandidateTarget* MemoryAgent::getBestTarget() const {
  if (candidates_.empty()) return nullptr;

  // Find candidate with lowest opportunity cost
  const CandidateTarget* best = nullptr;
  double best_cost = std::numeric_limits<double>::max();

  for (const auto& candidate : candidates_) {
    if (candidate.opportunity_cost < best_cost) {
      best_cost = candidate.opportunity_cost;
      best = &candidate;
    }
  }

  return best;
}

bool MemoryAgent::checkVerification(const Eigen::Vector2d& robot_pos, double robot_yaw) {
  last_robot_pos_ = robot_pos;

  // Single-shot verification: fire one VLM call per candidate when the robot
  // crosses verify_distance_. The Python verifier sends N annotated frames in
  // one call and returns a CONFIRM/UNCERTAIN/REJECT decision. No voting.
  const double trigger_dist = verify_distance_;

  for (size_t i = 0; i < candidates_.size(); ++i) {
    auto& candidate = candidates_[i];

    if (current_nav_target_ >= 0 && candidate.cluster_id != current_nav_target_) continue;

    double dist = (robot_pos - candidate.position).norm();
    if (dist > trigger_dist + 0.5) continue;
    if (!candidate.shouldVerifyAtDistance(dist, trigger_dist)) continue;
    // 2026-05-20: trust Qwen-VL more, fire on first cluster sighting.
    // observation_count starts at 1 on creation so `< 1` is effectively no gate.
    // The Python verifier filters frames to "target-detected" ones so even a
    // brand-new cluster's verification call gets fed proper annotated frames.
    if (candidate.observation_count < 1) continue;

    if (!checkThrottle()) {
      ROS_WARN_THROTTLE(5.0, "[MemoryAgent] VLM throttle limit reached");
      return false;
    }

    sendVLMRequest(i, plan_env::VLMMemoryRequest::REQUEST_VERIFY, robot_pos, robot_yaw);
    candidate.vlm_pending = true;
    ROS_INFO("[MemoryAgent] Single-shot verification fired for cluster %d at dist=%.2fm",
             candidate.cluster_id, dist);
    return true;
  }

  return false;
}

void MemoryAgent::setCurrentNavigationTarget(int cluster_id) {
  current_nav_target_ = cluster_id;

  // Find and reset verification state for this target
  auto it = cluster_to_candidate_.find(cluster_id);
  if (it != cluster_to_candidate_.end()) {
    candidates_[it->second].resetVerificationState();
    ROS_INFO("[MemoryAgent] Set navigation target to cluster %d, reset verification state", cluster_id);
  }
}

void MemoryAgent::sendVLMRequest(int target_idx, uint8_t request_type,
                                 const Eigen::Vector2d& robot_pos, double robot_yaw) {
  if (target_idx < 0 || target_idx >= static_cast<int>(candidates_.size())) return;

  const auto& candidate = candidates_[target_idx];

  plan_env::VLMMemoryRequest req;
  req.header.stamp = ros::Time::now();
  req.request_type = request_type;
  req.robot_x = robot_pos.x();
  req.robot_y = robot_pos.y();
  req.robot_yaw = robot_yaw;
  req.target_node_id = candidate.anchor_node_id;
  req.target_node_x = candidate.position.x();
  req.target_node_y = candidate.position.y();
  req.target_object = target_object_;
  req.target_cluster_id = candidate.cluster_id;
  req.nearby_memories = getNearbyMemoriesString(robot_pos);
  req.semantic_value = candidate.semantic_value;
  req.calls_this_minute = getVLMCallsThisMinute();

  vlm_request_pub_.publish(req);
  recordVLMCall();
  // Tell the async queue to expect one more response. Without this the queue
  // would treat the eventual provide() as a stale / unexpected response and
  // drop it on the floor.
  vlm_queue_.markFired();
}

void MemoryAgent::vlmResponseCallback(const plan_env::VLMMemoryResponse::ConstPtr& msg) {
  // ROS callback thread — parse only, do NOT touch candidates_ / topology
  // here. All state mutation happens on the agent loop in
  // applyPendingVerifications(), called from updateCandidates().
  CandidateVerificationResult result;
  result.target_node_id      = msg->target_node_id;
  result.verification_result = msg->verification_result;
  result.update_memory       = msg->update_memory;
  result.scene_description   = msg->scene_description;
  result.inferred_room_type  = msg->inferred_room_type;
  result.observed_objects.assign(msg->observed_objects.begin(),
                                 msg->observed_objects.end());
  result.confidence          = msg->confidence;
  vlm_queue_.provide(result);
}

void MemoryAgent::applyPendingVerifications() {
  CandidateVerificationResult result;
  while (vlm_queue_.tryConsume(result)) {
    // Route by anchor_node_id to find the candidate this response is for.
    int target_idx = -1;
    for (size_t i = 0; i < candidates_.size(); ++i) {
      if (candidates_[i].anchor_node_id == result.target_node_id) {
        target_idx = static_cast<int>(i);
        break;
      }
    }

    if (target_idx < 0) {
      // The candidate may have been merged away or evicted between request
      // fire and response arrival. Skip silently — markFired/provide pairing
      // already maintained queue accounting.
      ROS_WARN("[MemoryAgent] VLM response for unknown node %d (candidate evicted?)",
               result.target_node_id);
      continue;
    }

    auto& candidate = candidates_[target_idx];
    candidate.vlm_pending = false;
    candidate.verification_count++;
    candidate.last_verification = ros::Time::now();

    const char* result_str;
    switch (result.verification_result) {
      case plan_env::VLMMemoryResponse::VERIFY_FALSE_POSITIVE: result_str = "FALSE_POSITIVE"; break;
      case plan_env::VLMMemoryResponse::VERIFY_UNCERTAIN:      result_str = "UNCERTAIN";      break;
      case plan_env::VLMMemoryResponse::VERIFY_TRUE_POSITIVE:  result_str = "TRUE_POSITIVE";  break;
      default:                                                 result_str = "UNKNOWN";        break;
    }

    ROS_INFO("[MemoryAgent] Single-shot VLM decision for cluster %d: %s",
             candidate.cluster_id, result_str);

    // Update topology node memory (used as context for future LLM prompts).
    if (result.update_memory && topology_) {
      TopologyNode* node = topology_->getNodeById(result.target_node_id);
      if (node) {
        node->memory.scene_description = result.scene_description;
        node->memory.inferred_room_type = result.inferred_room_type;
        node->memory.observed_objects = result.observed_objects;
        node->memory.description_generated = true;
        node->memory.last_vlm_update = ros::Time::now().toSec();
        node->memory.memory_confidence = result.confidence;
        node->memory.verified = true;
      }
    }

    // Single-shot: the VLM call's decision is final. No aggregation needed.
    candidate.final_verification_result = result.verification_result;
    candidate.verification_complete = true;
    applyFinalVerificationResult(candidate);
  }
}

void MemoryAgent::applyFinalVerificationResult(CandidateTarget& candidate) {
  uint8_t result = candidate.final_verification_result;

  if (result == plan_env::VLMMemoryResponse::VERIFY_FALSE_POSITIVE) {
    // FALSE POSITIVE confirmed by multi-frame voting
    ROS_WARN("[MemoryAgent] === FINAL: FALSE POSITIVE for cluster %d ===", candidate.cluster_id);

    // 1. Record false positive
    recordFalsePositive(candidate.cluster_id);

    // 2. Reduce ObjectMap cluster confidence
    if (object_map_) {
      object_map_->adjustClusterConfidence(candidate.cluster_id, cluster_fp_penalty_);
    }

    // 3. Apply penalty to topology node
    if (topology_ && candidate.anchor_node_id >= 0) {
      TopologyNode* node = topology_->getNodeById(candidate.anchor_node_id);
      if (node) {
        node->applyFalsePosivePenalty(node_fp_penalty_strong_);
      }
    }
  }
  else if (result == plan_env::VLMMemoryResponse::VERIFY_TRUE_POSITIVE) {
    // TRUE POSITIVE confirmed by multi-frame voting
    ROS_INFO("[MemoryAgent] === FINAL: TRUE POSITIVE for cluster %d ===", candidate.cluster_id);

    // 1. Record success
    recordSuccess(candidate.cluster_id);

    // 2. Boost ObjectMap cluster confidence
    if (object_map_) {
      object_map_->adjustClusterConfidence(candidate.cluster_id, cluster_success_boost_);
    }

    // 3. Boost topology node semantic value
    if (topology_ && candidate.anchor_node_id >= 0) {
      TopologyNode* node = topology_->getNodeById(candidate.anchor_node_id);
      if (node) {
        node->semantic_multiplier =
            std::min(semantic_boost_cap_, node->semantic_multiplier * semantic_boost_factor_);
      }
    }
  }
  else {
    // UNCERTAIN - no change to confidence
    ROS_INFO("[MemoryAgent] === FINAL: UNCERTAIN for cluster %d (no action) ===", candidate.cluster_id);
  }

  // Clear navigation target after final decision
  if (current_nav_target_ == candidate.cluster_id) {
    current_nav_target_ = -1;
  }
}

bool MemoryAgent::queryVerificationNear(const Eigen::Vector2d& pos,
                                        double match_radius,
                                        bool& pending,
                                        bool& complete,
                                        bool& is_false_positive) const {
  pending = false;
  complete = false;
  is_false_positive = false;

  int best_idx = -1;
  double best_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < candidates_.size(); ++i) {
    double d = (candidates_[i].position - pos).norm();
    if (d < best_dist && d <= match_radius) {
      best_dist = d;
      best_idx = static_cast<int>(i);
    }
  }
  if (best_idx < 0) return false;

  const auto& c = candidates_[best_idx];
  pending = c.vlm_pending;
  complete = c.verification_complete;
  is_false_positive = (c.verification_complete &&
      c.final_verification_result == plan_env::VLMMemoryResponse::VERIFY_FALSE_POSITIVE);
  return true;
}

void MemoryAgent::recordSuccess(int cluster_id) {
  auto it = cluster_to_candidate_.find(cluster_id);
  if (it == cluster_to_candidate_.end()) return;

  auto& candidate = candidates_[it->second];
  candidate.cumulative_confidence =
      std::min(1.0, candidate.cumulative_confidence * confidence_success_boost_);

  // Boost topology node value
  if (topology_ && candidate.anchor_node_id >= 0) {
    TopologyNode* node = topology_->getNodeById(candidate.anchor_node_id);
    if (node) {
      node->semantic_multiplier =
          std::min(semantic_boost_cap_, node->semantic_multiplier * semantic_boost_factor_);
    }
  }

  ROS_INFO("[MemoryAgent] Recorded success for cluster %d", cluster_id);
}

void MemoryAgent::recordFalsePositive(int cluster_id) {
  auto it = cluster_to_candidate_.find(cluster_id);
  if (it == cluster_to_candidate_.end()) return;

  auto& candidate = candidates_[it->second];
  candidate.false_positive_count++;
  candidate.cumulative_confidence *= confidence_fp_penalty_;

  // Penalize topology node
  if (topology_ && candidate.anchor_node_id >= 0) {
    TopologyNode* node = topology_->getNodeById(candidate.anchor_node_id);
    if (node) {
      node->applyFalsePosivePenalty(node_fp_penalty_weak_);
      node->memory.false_positive_count++;
    }
  }

  ROS_WARN("[MemoryAgent] Recorded false positive for cluster %d (total: %d)",
           cluster_id, candidate.false_positive_count);
}

std::string MemoryAgent::getNearbyMemoriesString(const Eigen::Vector2d& pos) const {
  if (!topology_) return "";

  auto nearby = topology_->getNearbyNodesWithMemory(pos, memory_search_radius_, 3);

  std::string result;
  for (const auto& pair : nearby) {
    const TopologyNode* node = topology_->getNodeById(pair.first);
    if (!node || !node->hasMemory()) continue;

    if (!result.empty()) result += ";";

    // Format: "node_id:room_type:obj1,obj2,obj3"
    result += std::to_string(node->id) + ":" + node->memory.inferred_room_type + ":";

    for (size_t i = 0; i < node->memory.observed_objects.size(); ++i) {
      if (i > 0) result += ",";
      result += node->memory.observed_objects[i];
    }
  }

  return result;
}

bool MemoryAgent::checkThrottle() {
  double now = ros::Time::now().toSec();

  // Remove timestamps older than 1 minute
  while (!vlm_call_timestamps_.empty() &&
         now - vlm_call_timestamps_.front() > 60.0) {
    vlm_call_timestamps_.pop_front();
  }

  return static_cast<int>(vlm_call_timestamps_.size()) < max_vlm_calls_per_minute_;
}

void MemoryAgent::recordVLMCall() {
  vlm_call_timestamps_.push_back(ros::Time::now().toSec());
}

}  // namespace skillnav_planner
