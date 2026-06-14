#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// ROS message types
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <exploration_manager/agent_coordinator.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace skillnav_planner {
// Centralized constants for ExplorationFSM (mirrors the style of FSMConstants in fsm2.h)
namespace FSMConstants {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 0.25;

// Robot Action
constexpr double ACTION_DISTANCE = 0.25;
constexpr double ACTION_ANGLE = M_PI / 6.0;

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;       // consider stuck if movement < this
constexpr double REACH_DISTANCE = 0.20;          // reach object distance (kept tight; VLM gate fires here)
// 2026-05-21 (v2): lifted 0.60 → 0.85. Habitat's success criterion is
// distance_to_goal ≤ 1.0m. Since VLM-confirmed clusters sit on the actual
// target object, agent at 0.85m from cluster centroid is almost always within
// 1.0m of the goal viewpoint (offset typically 0.1-0.3m). When KinoAstar
// can't path closer, declaring REACH here saves the episode from stepout.
constexpr double SOFT_REACH_DISTANCE = 0.85;
constexpr double LOCAL_DISTANCE = 0.80;          // local target lookahead
constexpr double FORWARD_DISTANCE = 0.15;        // min clearance for marking obstacles
constexpr double FORCE_DORMANT_DISTANCE = 0.35;  // force dormant frontier if very close
constexpr double MIN_SAFE_DISTANCE = 0.15;       // min safe distance to obstacles

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;           // max consecutive stuck actions -> stop
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;  // times next_pos unchanged while stuck

// Adaptive Tryout constants (ETPNav-style escape)
// Tryout angles in turn counts: [-90°, -60°, -30°, +30°, +60°, +90°, 0°]
// Negative = RIGHT turns, Positive = LEFT turns (each turn is 30°)
constexpr int TRYOUT_TURN_COUNTS[7] = {-3, -2, -1, 1, 2, 3, 0};
constexpr int NUM_TRYOUT_ANGLES = 7;

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;  // penalize moving away
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;   // encourage moving closer
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;  // samples along a step for safety cost

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;  // multiply by map resolution
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;

// Sticky-commitment: anti-thrash for frontier selection.
// Once a target is selected, suppress switching to a different one unless
// (a) the new candidate is more than STICKY_RADIUS away, OR
// (b) we've been locked in for more than MAX_STICKY_STEPS, OR
// (c) a dormant / replan flag was raised by another subsystem.
constexpr double STICKY_RADIUS = 1.0;     // m; ~4 grid cells at 0.25 m resolution
// 2026-05-21: bumped 30 → 90. At 10 Hz that's ~9 s of commitment instead of 3 s,
// giving the agent time to actually travel 2-3 m toward the chosen target
// before the planner is allowed to flip. Reduces frontier-flip oscillation
// (visible as the agent zig-zagging between two near-equal candidates).
constexpr int MAX_STICKY_STEPS = 90;       // cycles before forced re-evaluation
}  // namespace FSMConstants

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum ROS_STATE { INIT, WAIT_TRIGGER, PLAN_ACTION, WAIT_ACTION_FINISH, PUB_ACTION, FINISH };
enum ACTION { STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_DOWN, TURN_UP };
enum HABITAT_STATE { READY, ACTION_EXEC, ACTION_FINISH, EPISODE_FINISH };
class ExplorationFSM {
private:
  /* Planning Utils */
  ros::NodeHandle nh_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  ROS_STATE state_;

  /* ROS Utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, habitat_state_sub_, confidence_threshold_sub_;
  ros::Publisher action_pub_, ros_state_pub_, expl_state_pub_, expl_result_pub_;
  ros::Publisher robot_marker_pub_;




  /* AgentCoordinator — single owner of the per-time-scale agents.
     Currently owns SafeAgent and views MemoryAgent (which ExplorationManager
     still owns). Future home of StrategicAgent + AsyncResultBuffer<T>. See
     exploration_manager/agent_coordinator.{h,cpp}. */
  std::unique_ptr<AgentCoordinator> agents_;

  /* Action Planner */
  int callActionPlanner();
  int planNextBestAction(Vector2d current_pos, double current_yaw, const vector<Vector2d>& path,
      bool need_safety = true);
  Vector2d selectLocalTarget(
      const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance);
  int decideNextAction(double current_yaw, double target_yaw);
  Vector2d computeBestStep(
      const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos);
  double computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step);
  double computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
      const Vector2d& target_pos, const Vector2d& step);

  /* Helper functions */
  bool updateFrontierAndObject();
  void transitState(ROS_STATE new_state, string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS callbacks */
  void FSMCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32ConstPtr& msg);
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);

  /* SafeAgent helpers moved to exploration_manager/safe_agent.{h,cpp} — keep
     this section empty for now to make the migration visible in diff. */

public:
  ExplorationFSM() = default;
  ~ExplorationFSM() = default;

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSM::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}
}  // namespace skillnav_planner

#endif