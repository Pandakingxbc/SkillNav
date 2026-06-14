#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <trajectory_manager/optimizer.h>

// Undefine uint macro from optimizer.h to avoid conflict with OpenCV
#ifdef uint
#undef uint
#endif

namespace skillnav_planner {

enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };

/**
 * @brief Adaptive Tryout state machine for escape-from-stuck
 *
 * Inspired by ETPNav's Tryout mechanism:
 * - Try 7 angles: [-90°, -60°, -30°, +30°, +60°, +90°, 0°]
 * - For each angle: rotate, try forward, check if moved
 * - If success: restore original heading
 * - If all fail: mark obstacle
 */
enum TryoutState {
  TRYOUT_IDLE,           ///< Not in tryout mode
  TRYOUT_ROTATING,       ///< Rotating to try angle
  TRYOUT_FORWARD,        ///< Trying forward movement
  TRYOUT_RESTORE,        ///< Restoring original heading after success
  TRYOUT_FAILED          ///< All angles tried, marking obstacle
};

struct FSMData {
  FSMData()
  {
    trigger_ = false;
    have_odom_ = false;
    have_confidence_ = false;
    have_finished_ = false;
    static_state_ = true;
    state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_ACTION", "WAIT_ACTION_FINISH", "PUB_ACTION",
      "FINISH" };

    odom_pos_ = Eigen::Vector3d::Zero();
    odom_vel_ = Eigen::Vector3d::Zero();
    odom_omega_ = Eigen::Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Eigen::Vector3d::Zero();
    start_vel_ = Eigen::Vector3d::Zero();
    start_yaw_ = Eigen::Vector3d::Zero();
    last_start_pos_ = Eigen::Vector3d(-100, -100, -100);
    last_next_pos_ = Eigen::Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    sticky_step_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    // Adaptive Tryout state
    tryout_state_ = TRYOUT_IDLE;
    tryout_angle_index_ = 0;
    tryout_turns_remaining_ = 0;
    tryout_original_yaw_ = 0.0;
    tryout_pre_forward_pos_ = Eigen::Vector2d::Zero();
    tryout_restore_turns_ = 0;

    local_pos_ = Eigen::Vector2d(0, 0);
  }
  // FSM data
  bool trigger_, have_odom_, have_confidence_;
  bool have_finished_;
  std::vector<string> state_str_;
  std::vector<Eigen::Vector2d> traveled_path_;

  // odometry state
  Eigen::Vector3d odom_pos_, odom_vel_, odom_omega_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;
  bool static_state_;  // Track if robot is static or moving

  Eigen::Vector3d start_pt_, start_vel_, start_yaw_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;
  /// Sticky-commitment counter: number of consecutive replans that have
  /// reused the previously-committed next_pos_ instead of switching to a
  /// freshly-computed candidate. Reset to 0 whenever a new commitment is
  /// made or the lock is released. See ExplorationFSM::callActionPlanner.
  int sticky_step_count_;

  int final_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Eigen::Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  std::vector<Eigen::Vector3d> stucking_points_;

  // Adaptive Tryout state (ETPNav-style escape mechanism)
  TryoutState tryout_state_;
  int tryout_angle_index_;        ///< Current angle being tried (0-6)
  int tryout_turns_remaining_;    ///< Turns left to reach target angle
  double tryout_original_yaw_;    ///< Original heading to restore after success
  Eigen::Vector2d tryout_pre_forward_pos_;  ///< Position before trying forward
  int tryout_restore_turns_;      ///< Turns needed to restore heading
  bool tryout_turn_direction_;    ///< true=LEFT, false=RIGHT for current rotation

  Eigen::Vector2d local_pos_;
  LocalTrajectory newest_traj_;  // Store latest planned trajectory
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;
    replan_time_ = 0.2;
    replan_traj_end_threshold_ = 1.0;
    replan_frontier_change_delay_ = 0.5;
    replan_timeout_ = 2.0;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  std::vector<Eigen::Vector2d> action_steps_;
  // replan timing parameters (loaded from ros params in ExplorationFSM::init)
  double replan_time_;
  double replan_traj_end_threshold_;
  double replan_frontier_change_delay_;
  double replan_timeout_;
};

struct ExplorationData {
  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Eigen::Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
  }
  std::vector<std::vector<Eigen::Vector2d>> frontiers_, dormant_frontiers_;
  std::vector<Eigen::Vector2d> frontier_averages_, dormant_frontier_averages_;
  std::vector<std::vector<Eigen::Vector2d>> objects_;
  std::vector<Eigen::Vector2d> object_averages_;
  std::vector<int> object_labels_;
  Eigen::Vector2d next_pos_;
  Eigen::Vector2d next_local_pos_;  // Local target position along path
  std::vector<Eigen::Vector2d> next_best_path_;
  std::vector<Eigen::Vector2d> tsp_tour_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST, HFTN };
  // params
  int policy_mode_;
  double sigma_threshold_, max_to_mean_threshold_, max_to_mean_percentage_;
  std::string tsp_dir_;

  // C1 Hybrid Frontier-Topology (HFTN) policy params. The node score is
  //   V_node = (anchored_ig_sum / hftn_ig_scale)
  //         + hftn_degree_bonus * log(1 + degree)
  //         + hftn_base_weight  * base_value
  // multiplied by the node's passability & semantic multipliers. Loaded
  // from rosparam in ExplorationManager::initialize so they can be set
  // per-launch in algorithm.xml.
  double hftn_degree_bonus_;   ///< β — coefficient on log(1+degree)
  double hftn_base_weight_;    ///< γ — coefficient on MVMM base_value at node
  double hftn_ig_scale_;       ///< Divisor turning anchored cell count into [0,1]-ish range
};

}  // namespace skillnav_planner

#endif
