#pragma once

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <memory>
#include <string>

#include <plan_env/VLMDeadZoneResponse.h>

#include <plan_env/async_result_buffer.h>

namespace skillnav_planner {

/**
 * Result of one asynchronous VLM dead-zone consultation. SafeAgent's response
 * callback parses the raw ROS message into this struct and hands it to the
 * AsyncResultBuffer; the agent loop later picks it up and applies the effect
 * to the relevant Voronoi node. Keeping ROS message types out of the agent
 * loop keeps the loop testable without bringing up a ROS master.
 */
struct DeadZoneVLMResult {
  int target_node_id{-1};
  std::string penalty_strength;   ///< "strong" | "moderate" | "weak" | "none"
  std::string obstacle_type;      ///< "permanent" | "temporary" | "system_boundary" | "unclear"
  double confidence{0.0};
  double retry_delay_seconds{0.0};
};

// Forward declarations to keep this header lightweight.
struct FSMData;
class ExplorationManager;

/**
 * SafeAgent — Reactive layer for stuck recovery in SkillNav.
 *
 * Owns the adaptive Tryout state machine (ETPNav-style 7-angle probing) and the
 * asynchronous VLM Dead-Zone consultation that decides whether a stuck region is
 * a permanent obstacle, a temporary blockage, a system boundary, or unclear.
 *
 * Time scale: reactive — invoked from the FSM's ~10 Hz callActionPlanner before
 * normal frontier planning. Never blocks; VLM consultation is fire-and-forget
 * and its response asynchronously updates the relevant Voronoi node's
 * passability multiplier (consumed by frontier scoring on subsequent ticks).
 *
 * Ownership: constructed once per ExplorationFSM. Holds non-owning shared_ptrs
 * to FSMData and ExplorationManager so it can mutate the FSM's stuck-tracking
 * fields (escape_stucking_*, tryout_*, stucking_points_) and write to SDFMap /
 * VoronoiTopology. This keeps the refactor minimal-invasive: state stays where
 * FSMConstants and the action enum already see it.
 *
 * Future evolution: when the AgentCoordinator + AsyncResultBuffer pattern lands,
 * the synchronous publisher/subscriber pair here is replaced with a
 * Promise/Future wrapping the same /safe_agent/vlm_request|vlm_response topics.
 */
/**
 * Outcome of one SafeAgent::tick() invocation.
 *
 * The FSM examines this to decide whether to bypass its normal frontier-planning
 * logic for the current 10 Hz cycle, treat the episode as terminated, or proceed
 * with planning as usual.
 */
enum class SafeTickResult {
  NotEscaping,    ///< Robot is not stuck; FSM should proceed with normal planning.
  EmittedAction,  ///< Tryout state machine emitted an action; FSM should return
                  ///< fd_->final_result_ unchanged for this tick.
  ReachedObject,  ///< Stuck-detection short-circuited because robot is already
                  ///< within soft-reach distance of a SEARCH_OBJECT target;
                  ///< FSM should return REACH_OBJECT for this tick.
};

class SafeAgent {
public:
  SafeAgent() = default;
  ~SafeAgent() = default;

  /// One-time init: bind to dependencies, advertise / subscribe VLM topics,
  /// load penalty policy params from rosparam.
  void init(ros::NodeHandle& nh,
            std::shared_ptr<FSMData> fd,
            std::shared_ptr<ExplorationManager> em);

  /**
   * Main entry called from FSM each tick BEFORE normal planning.
   *
   * Sequence:
   *   1. If not yet escaping, detect new stuck (FORWARD that did not move).
   *      Short-circuit to ReachedObject if already at SEARCH_OBJECT target.
   *   2. If newly escaping and Tryout is IDLE, initialize first angle.
   *   3. Advance Tryout state machine; possibly emit an action.
   *   4. On TRYOUT_FAILED: mark forceOccGrid + dormant frontier flag + record
   *      stuck point + fire async VLM consultation, then return NotEscaping so
   *      the FSM re-plans around the newly-marked obstacles.
   */
  /// last_pos MUST be the snapshot of fd_->last_start_pos_ captured BEFORE the
  /// FSM overwrites it with the current pose (which happens at the top of
  /// callActionPlanner). Reading fd_->last_start_pos_ inside SafeAgent itself
  /// is unsafe because the FSM updates it before this tick() is invoked,
  /// which silently makes every cycle look like a stuck event.
  SafeTickResult tick(const Eigen::Vector2d& current_pos, double current_yaw,
                      const Eigen::Vector2d& last_pos,
                      double stucking_distance, double soft_reach_distance);

  /// Whether the Tryout state machine is currently active.
  bool isEscaping() const;

  /// ROS callback — VLM judgment of a previously-requested dead zone.
  void onVLMResponse(const plan_env::VLMDeadZoneResponse::ConstPtr& msg);

private:
  /// Three outcomes mirroring SafeTickResult — used internally by tick().
  enum class DetectOutcome { NoNewStuck, NewStuckInitialized, ReachedObject };

  /// Inspect motion since last tick and either declare reach-object, initialize
  /// a new Tryout episode, or no-op. Mutates fd_->escape_stucking_* on init.
  DetectOutcome detectAndInitStuck(const Eigen::Vector2d& current_pos,
                                   double current_yaw,
                                   const Eigen::Vector2d& last_pos,
                                   double stucking_distance,
                                   double soft_reach_distance);

  /// Advances the 4-state Tryout state machine; returns true if action emitted
  /// (i.e. caller should treat this as EmittedAction). Returns false only when
  /// TRYOUT_FAILED was just handled (obstacles marked, VLM consulted) so the
  /// FSM should fall through to normal planning.
  bool runTryoutStateMachine(const Eigen::Vector2d& current_pos,
                             double stucking_distance);

  /// Called once when all 7 angles failed: mark obstacle, record stuck point, fire VLM.
  void onTryoutFailed(const Eigen::Vector2d& current_pos);

  /// Asynchronously request VLM dead-zone judgment.
  void requestVLMDeadZone(const Eigen::Vector2d& stuck_pos, double stuck_yaw,
                          int escape_attempts, int angles_tried);

  /// Helper: nearest Voronoi node id, or -1 if topology not available.
  int findNearestNodeId(const Eigen::Vector2d& pos);

  /// Drain `vlm_buffer_` and apply the result to the associated Voronoi node.
  /// Called at the top of each tick(), before any escape-state work.
  void applyPendingVLMResult();

  // ── Non-owning dependencies ─────────────────────────────────────────
  std::shared_ptr<FSMData> fd_;
  std::shared_ptr<ExplorationManager> em_;

  // ── VLM async state ─────────────────────────────────────────────────
  ros::Publisher  vlm_request_pub_;
  ros::Subscriber vlm_response_sub_;
  /// Single-slot mailbox for VLM dead-zone judgments. Filled by the ROS
  /// callback (`onVLMResponse`), drained by `tick()` once per cycle.
  AsyncResultBuffer<DeadZoneVLMResult> vlm_buffer_;
  /// Voronoi node id associated with the in-flight request. Kept alongside the
  /// buffer (rather than in the result struct) so it survives even if the VLM
  /// response forgets to echo it. Cleared on cancel / consume.
  int pending_node_id_{-1};

  // ── Penalty policy (loaded from rosparam in init) ───────────────────
  double penalty_strong_{0.1};        ///< → VLM_CONFIRMED_DEAD
  double penalty_moderate_{0.3};      ///< → HIGH_RISK
  double penalty_weak_{0.6};          ///< → TEMPORARILY_BLOCKED
  double retry_delay_default_{30.0};  ///< s, fallback when VLM omits retry hint

  /// Number of Tryout angles to attempt (default = full 7-angle adaptive).
  /// Set to 1 via `safe_agent/num_tryout_angles=1` for "simple escape"
  /// ablation — agent tries the first angle (-90°) and gives up if it fails,
  /// which dramatically increases stuck-related failures.
  int num_tryout_angles_{7};
};

}  // namespace skillnav_planner
