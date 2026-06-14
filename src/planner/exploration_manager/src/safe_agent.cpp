#include <exploration_manager/safe_agent.h>

#include <cmath>

#include <exploration_manager/exploration_data.h>     // FSMData, TryoutState, FINAL_RESULT
#include <exploration_manager/exploration_fsm.h>      // ACTION enum, FSMConstants
#include <exploration_manager/exploration_manager.h>  // ExplorationManager
#include <plan_env/sdf_map2d.h>                       // setForceOccGrid
#include <plan_env/voronoi_topology.h>                // VoronoiTopology, DeadZoneStatus
#include <plan_env/VLMDeadZoneRequest.h>

namespace skillnav_planner {

void SafeAgent::init(ros::NodeHandle& nh,
                     std::shared_ptr<FSMData> fd,
                     std::shared_ptr<ExplorationManager> em)
{
  fd_ = fd;
  em_ = em;
  pending_node_id_ = -1;

  vlm_request_pub_ = nh.advertise<plan_env::VLMDeadZoneRequest>(
      "/safe_agent/vlm_request", 10);
  vlm_response_sub_ = nh.subscribe(
      "/safe_agent/vlm_response", 10, &SafeAgent::onVLMResponse, this);

  nh.param("safe_agent/penalty_strong",      penalty_strong_,      0.1);
  nh.param("safe_agent/penalty_moderate",    penalty_moderate_,    0.3);
  nh.param("safe_agent/penalty_weak",        penalty_weak_,        0.6);
  nh.param("safe_agent/retry_delay_default", retry_delay_default_, 30.0);
  nh.param("safe_agent/num_tryout_angles",   num_tryout_angles_,
           FSMConstants::NUM_TRYOUT_ANGLES);
  if (num_tryout_angles_ < 1 || num_tryout_angles_ > FSMConstants::NUM_TRYOUT_ANGLES) {
    ROS_WARN("[SafeAgent] num_tryout_angles=%d out of [1, %d], clamping",
             num_tryout_angles_, FSMConstants::NUM_TRYOUT_ANGLES);
    num_tryout_angles_ = std::max(1, std::min(num_tryout_angles_,
                                              FSMConstants::NUM_TRYOUT_ANGLES));
  }
  ROS_INFO("[SafeAgent] init: num_tryout_angles=%d (%s)",
           num_tryout_angles_,
           num_tryout_angles_ == FSMConstants::NUM_TRYOUT_ANGLES
               ? "full adaptive" : "ABLATION: simple escape");
}

bool SafeAgent::isEscaping() const {
  return fd_ && fd_->escape_stucking_flag_;
}

SafeTickResult SafeAgent::tick(const Eigen::Vector2d& current_pos,
                               double current_yaw,
                               const Eigen::Vector2d& last_pos,
                               double stucking_distance,
                               double soft_reach_distance)
{
  // Phase 0: drain any pending VLM result that arrived since last tick. This
  // is the single authoritative place where async judgments touch Voronoi
  // state, keeping the ROS callback thread free of map mutations.
  applyPendingVLMResult();

  // Phase 1: new stuck detection only when not already in an escape episode.
  if (!fd_->escape_stucking_flag_) {
    const auto outcome = detectAndInitStuck(current_pos, current_yaw, last_pos,
                                            stucking_distance, soft_reach_distance);
    if (outcome == DetectOutcome::ReachedObject)  return SafeTickResult::ReachedObject;
    if (outcome == DetectOutcome::NoNewStuck)     return SafeTickResult::NotEscaping;
    // NewStuckInitialized falls through with escape_stucking_flag_ = true,
    // tryout_state_ = TRYOUT_IDLE.
  }

  // Phase 2: lazy Tryout init on the first escaping tick.
  if (fd_->escape_stucking_flag_ && fd_->tryout_state_ == TRYOUT_IDLE) {
    fd_->tryout_state_ = TRYOUT_ROTATING;
    fd_->tryout_angle_index_ = 0;
    fd_->tryout_original_yaw_ = fd_->escape_stucking_yaw_;
    const int turn_count = FSMConstants::TRYOUT_TURN_COUNTS[0];
    fd_->tryout_turns_remaining_ = std::abs(turn_count);
    fd_->tryout_turn_direction_ = (turn_count > 0);
    ROS_WARN("[SafeAgent/Tryout] Starting adaptive escape, angle %d (%d turns %s)",
             fd_->tryout_angle_index_, fd_->tryout_turns_remaining_,
             fd_->tryout_turn_direction_ ? "LEFT" : "RIGHT");
  }

  // Phase 3: state machine.
  if (fd_->escape_stucking_flag_ && fd_->tryout_state_ != TRYOUT_IDLE) {
    if (runTryoutStateMachine(current_pos, stucking_distance)) {
      return SafeTickResult::EmittedAction;
    }
    // runTryoutStateMachine returned false → TRYOUT_FAILED handled already,
    // fall through so FSM replans around the newly-marked obstacles.
  }
  return SafeTickResult::NotEscaping;
}

SafeAgent::DetectOutcome SafeAgent::detectAndInitStuck(
    const Eigen::Vector2d& current_pos, double current_yaw,
    const Eigen::Vector2d& last_pos,
    double stucking_distance, double soft_reach_distance)
{
  // last_pos was snapshotted by the FSM BEFORE fd_->last_start_pos_ was
  // overwritten with the current position; reading fd_->last_start_pos_ here
  // would always yield current_pos and make every cycle look stuck.
  const int last_action = fd_->newest_action_;
  const double moved_dist = (current_pos - last_pos).norm();

  // REVERTED to ApexNav-original strictness (2026-05-21 22:00 — my earlier
  // relaxation triggered a Tryout-SUCCESS-Tryout death loop, eps ballooning
  // from 50-100s to 1700-2400s). Stuck only when actively trying to forward.
  if (moved_dist >= stucking_distance || last_action != ACTION::MOVE_FORWARD) {
    return DetectOutcome::NoNewStuck;
  }

  // Stuck while right at a SEARCH_OBJECT target: declare success instead of
  // launching an escape attempt.
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - em_->ed_->next_pos_).norm() < soft_reach_distance) {
    ROS_ERROR("Reach the object successfully!!!");
    return DetectOutcome::ReachedObject;
  }

  // Skip if this exact (position, yaw) has already failed an escape — prevents
  // re-entering a known dead state.
  for (const auto& sp : fd_->stucking_points_) {
    const Eigen::Vector2d stucking_pos(sp(0), sp(1));
    const double stucking_yaw = sp(2);
    if ((stucking_pos - current_pos).norm() < stucking_distance &&
        std::fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
      ROS_ERROR("Still stuck at the same place");
      return DetectOutcome::NoNewStuck;
    }
  }

  fd_->escape_stucking_flag_ = true;
  fd_->escape_stucking_count_ = 0;
  fd_->escape_stucking_pos_ = current_pos;
  fd_->escape_stucking_yaw_ = current_yaw;
  fd_->tryout_state_ = TRYOUT_IDLE;
  fd_->tryout_angle_index_ = 0;
  return DetectOutcome::NewStuckInitialized;
}

bool SafeAgent::runTryoutStateMachine(const Eigen::Vector2d& current_pos,
                                      double stucking_distance)
{
  switch (fd_->tryout_state_) {
    case TRYOUT_ROTATING: {
      if (fd_->tryout_turns_remaining_ > 0) {
        fd_->newest_action_ = fd_->tryout_turn_direction_ ? ACTION::TURN_LEFT
                                                          : ACTION::TURN_RIGHT;
        fd_->tryout_turns_remaining_--;
        ROS_INFO("[SafeAgent/Tryout] Rotating: %d turns remaining",
                 fd_->tryout_turns_remaining_);
      } else {
        fd_->tryout_state_ = TRYOUT_FORWARD;
        fd_->tryout_pre_forward_pos_ = current_pos;
        fd_->newest_action_ = ACTION::MOVE_FORWARD;
        ROS_INFO("[SafeAgent/Tryout] Rotation done, trying FORWARD at angle %d",
                 fd_->tryout_angle_index_);
      }
      return true;
    }

    case TRYOUT_FORWARD: {
      const bool moved =
          (current_pos - fd_->tryout_pre_forward_pos_).norm() >= stucking_distance;
      if (moved) {
        ROS_WARN("[SafeAgent/Tryout] SUCCESS at angle %d! Restoring heading...",
                 fd_->tryout_angle_index_);
        fd_->tryout_state_ = TRYOUT_RESTORE;
        const int turn_count =
            FSMConstants::TRYOUT_TURN_COUNTS[fd_->tryout_angle_index_];
        fd_->tryout_restore_turns_ = std::abs(turn_count);
        fd_->tryout_turn_direction_ = (turn_count < 0);  // opposite of original
        if (fd_->tryout_restore_turns_ > 0) {
          fd_->newest_action_ = fd_->tryout_turn_direction_ ? ACTION::TURN_LEFT
                                                            : ACTION::TURN_RIGHT;
          fd_->tryout_restore_turns_--;
        } else {
          // Angle was 0° — already restored.
          fd_->escape_stucking_flag_ = false;
          fd_->tryout_state_ = TRYOUT_IDLE;
          ROS_WARN("[SafeAgent/Tryout] Escaped successfully!");
        }
        return true;
      }

      // Failed at this angle, advance to next.
      fd_->tryout_angle_index_++;
      if (fd_->tryout_angle_index_ < num_tryout_angles_) {
        const int turn_count =
            FSMConstants::TRYOUT_TURN_COUNTS[fd_->tryout_angle_index_];
        const int prev_turn_count =
            FSMConstants::TRYOUT_TURN_COUNTS[fd_->tryout_angle_index_ - 1];
        const int relative_turns = turn_count - prev_turn_count;
        fd_->tryout_turns_remaining_ = std::abs(relative_turns);
        fd_->tryout_turn_direction_ = (relative_turns > 0);
        fd_->tryout_state_ = TRYOUT_ROTATING;
        ROS_INFO("[SafeAgent/Tryout] Angle %d failed, trying angle %d (%d %s turns)",
                 fd_->tryout_angle_index_ - 1, fd_->tryout_angle_index_,
                 fd_->tryout_turns_remaining_,
                 fd_->tryout_turn_direction_ ? "LEFT" : "RIGHT");
      } else {
        fd_->tryout_state_ = TRYOUT_FAILED;
        ROS_ERROR("[SafeAgent/Tryout] All %d angles tried, escape FAILED!",
                  num_tryout_angles_);
      }
      return true;
    }

    case TRYOUT_RESTORE: {
      if (fd_->tryout_restore_turns_ > 0) {
        fd_->newest_action_ = fd_->tryout_turn_direction_ ? ACTION::TURN_LEFT
                                                          : ACTION::TURN_RIGHT;
        fd_->tryout_restore_turns_--;
        ROS_INFO("[SafeAgent/Tryout] Restoring heading: %d turns remaining",
                 fd_->tryout_restore_turns_);
      } else {
        fd_->escape_stucking_flag_ = false;
        fd_->tryout_state_ = TRYOUT_IDLE;
        ROS_WARN("[SafeAgent/Tryout] Heading restored. Escaped successfully!");
      }
      return true;
    }

    case TRYOUT_FAILED: {
      onTryoutFailed(current_pos);
      return false;  // signal FSM to replan
    }

    default:
      return false;
  }
}

void SafeAgent::onTryoutFailed(const Eigen::Vector2d& current_pos)
{
  ROS_ERROR("[SafeAgent] All angles failed. Marking obstacle + VLM consult.");
  fd_->escape_stucking_flag_ = false;
  fd_->tryout_state_ = TRYOUT_IDLE;

  // Three-cell obstacle stripe along the stuck heading.
  em_->sdf_map_->setForceOccGrid(current_pos);

  Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
  forward_pos(0) += FSMConstants::FORWARD_DISTANCE * std::cos(fd_->escape_stucking_yaw_);
  forward_pos(1) += FSMConstants::FORWARD_DISTANCE * std::sin(fd_->escape_stucking_yaw_);
  em_->sdf_map_->setForceOccGrid(forward_pos);

  forward_pos = fd_->escape_stucking_pos_;
  forward_pos(0) += (FSMConstants::FORWARD_DISTANCE * 2.0) * std::cos(fd_->escape_stucking_yaw_);
  forward_pos(1) += (FSMConstants::FORWARD_DISTANCE * 2.0) * std::sin(fd_->escape_stucking_yaw_);
  em_->sdf_map_->setForceOccGrid(forward_pos);

  fd_->dormant_frontier_flag_ = true;

  Eigen::Vector3d sp(fd_->escape_stucking_pos_(0),
                     fd_->escape_stucking_pos_(1),
                     fd_->escape_stucking_yaw_);
  fd_->stucking_points_.push_back(sp);

  requestVLMDeadZone(fd_->escape_stucking_pos_,
                     fd_->escape_stucking_yaw_,
                     fd_->escape_stucking_count_,
                     FSMConstants::NUM_TRYOUT_ANGLES);
}

void SafeAgent::requestVLMDeadZone(const Eigen::Vector2d& stuck_pos, double stuck_yaw,
                                   int escape_attempts, int angles_tried)
{
  if (vlm_buffer_.isInflight()) {
    ROS_WARN("[SafeAgent] VLM request already pending, skipping...");
    return;
  }

  const int node_id = findNearestNodeId(stuck_pos);
  pending_node_id_ = node_id;

  plan_env::VLMDeadZoneRequest req;
  req.header.stamp = ros::Time::now();
  req.header.frame_id = "world";
  req.robot_x = stuck_pos.x();
  req.robot_y = stuck_pos.y();
  req.robot_yaw = stuck_yaw;
  req.escape_attempt_count = escape_attempts;
  req.tryout_angles_tried = angles_tried;
  req.target_node_id = node_id;

  if (node_id >= 0 && em_->sdf_map_->voronoi_topology_) {
    auto* node = em_->sdf_map_->voronoi_topology_->getNearestNode(stuck_pos);
    if (node) {
      req.target_node_x = node->position.x();
      req.target_node_y = node->position.y();
    }
  }

  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT) {
    req.target_description = "navigating to detected object";
  } else if (fd_->final_result_ == FINAL_RESULT::EXPLORE) {
    req.target_description = "exploring frontier";
  } else {
    req.target_description = "unknown navigation task";
  }

  vlm_request_pub_.publish(req);
  vlm_buffer_.markFired();

  ROS_WARN("[SafeAgent] VLM dead zone analysis requested for node %d at (%.2f, %.2f)",
           node_id, stuck_pos.x(), stuck_pos.y());
}

void SafeAgent::onVLMResponse(const plan_env::VLMDeadZoneResponse::ConstPtr& msg)
{
  // ROS callback thread — keep this dead simple: parse and deposit. All side
  // effects on the Voronoi map happen later, on the agent loop, via
  // applyPendingVLMResult().
  if (!vlm_buffer_.isInflight()) {
    ROS_WARN("[SafeAgent] Received VLM response but no request pending, ignoring...");
    return;
  }

  DeadZoneVLMResult result;
  result.target_node_id      = pending_node_id_;
  result.penalty_strength    = msg->penalty_strength;
  result.obstacle_type       = msg->obstacle_type;
  result.confidence          = msg->confidence;
  result.retry_delay_seconds = msg->retry_delay_seconds;

  vlm_buffer_.provide(result);

  ROS_INFO("[SafeAgent] VLM response queued: type=%s, conf=%.2f, penalty=%s",
           msg->obstacle_type.c_str(), msg->confidence,
           msg->penalty_strength.c_str());
}

void SafeAgent::applyPendingVLMResult()
{
  DeadZoneVLMResult r;
  if (!vlm_buffer_.tryConsume(r)) return;

  if (r.target_node_id < 0 || !em_ || !em_->sdf_map_ ||
      !em_->sdf_map_->voronoi_topology_) {
    pending_node_id_ = -1;
    return;
  }

  auto& nodes = em_->sdf_map_->voronoi_topology_->getNodes();
  for (auto& node : nodes) {
    if (node.id != r.target_node_id) continue;

    double penalty_multiplier = 1.0;
    DeadZoneStatus new_status = DZ_NORMAL;

    if (r.penalty_strength == "strong") {
      penalty_multiplier = penalty_strong_;
      new_status = DZ_VLM_CONFIRMED_DEAD;
      ROS_WARN("[SafeAgent] Node %d → VLM_CONFIRMED_DEAD", r.target_node_id);
    } else if (r.penalty_strength == "moderate") {
      penalty_multiplier = penalty_moderate_;
      new_status = DZ_HIGH_RISK;
      ROS_WARN("[SafeAgent] Node %d → HIGH_RISK", r.target_node_id);
    } else if (r.penalty_strength == "weak") {
      penalty_multiplier = penalty_weak_;
      new_status = DZ_TEMPORARILY_BLOCKED;
      const double retry_delay = r.retry_delay_seconds > 0
                                     ? r.retry_delay_seconds
                                     : retry_delay_default_;
      node.retry_time = ros::Time::now().toSec() + retry_delay;
      ROS_WARN("[SafeAgent] Node %d → TEMPORARILY_BLOCKED, retry in %.1fs",
               r.target_node_id, retry_delay);
    } else {
      ROS_INFO("[SafeAgent] Node %d: no penalty applied (VLM uncertain)",
               r.target_node_id);
    }

    node.passability_multiplier *= penalty_multiplier;
    node.dead_zone_status = new_status;
    node.escape_fail_count++;

    ROS_INFO("[SafeAgent] Node %d updated: passability=%.3f, status=%d",
             r.target_node_id, node.passability_multiplier,
             static_cast<int>(node.dead_zone_status));
    break;
  }

  pending_node_id_ = -1;
}

int SafeAgent::findNearestNodeId(const Eigen::Vector2d& pos)
{
  if (!em_ || !em_->sdf_map_ || !em_->sdf_map_->voronoi_topology_) return -1;
  auto* node = em_->sdf_map_->voronoi_topology_->getNearestNode(pos);
  return node ? node->id : -1;
}

}  // namespace skillnav_planner
