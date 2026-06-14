#pragma once

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <memory>

#include <exploration_manager/safe_agent.h>
#include <exploration_manager/strategic_agent.h>

namespace skillnav_planner {

// Forward declarations to keep this header light.
struct FSMData;
class ExplorationManager;
class MemoryAgent;

/**
 * AgentCoordinator — single owner of SkillNav's per-time-scale agents.
 *
 * Why this class exists:
 *   Reviewers reading the paper see "we use a three-agent architecture" and
 *   want to point to a class in the codebase. Before this refactor SafeAgent
 *   lived as a unique_ptr on the FSM and MemoryAgent lived on the Exploration-
 *   Manager — there was no single place to look. AgentCoordinator is that
 *   place. It also pre-builds the seam where AsyncResultBuffer<T> and the
 *   StrategicAgent will plug in.
 *
 * Current ownership:
 *   - SafeAgent      : owned here (moved from ExplorationFSM).
 *   - StrategicAgent : owned here (newly built, v0 with stubbed LLM).
 *   - MemoryAgent    : viewed here (raw pointer); owned by ExplorationManager.
 *                      Migrating ownership upward is a follow-up; today's goal
 *                      is the structural seam, not the cleanup of every
 *                      call-site.
 *
 * Lifetime:
 *   Owned by ExplorationFSM. Constructed in FSM::init() after Exploration-
 *   Manager has finished initialize() so the MemoryAgent view is valid.
 */
class AgentCoordinator {
 public:
  AgentCoordinator() = default;
  ~AgentCoordinator() = default;

  /**
   * Wire up agents. After this returns, safe() is a valid SafeAgent pointer
   * and memory() returns the EM-owned MemoryAgent view (may be null if EM
   * skipped MemoryAgent construction).
   */
  void init(ros::NodeHandle& nh,
            std::shared_ptr<FSMData> fd,
            std::shared_ptr<ExplorationManager> em);

  /// Forward to SafeAgent::tick. Called from FSM each cycle before normal
  /// planning. last_pos must be the FSM's pre-overwrite snapshot of
  /// fd_->last_start_pos_; see SafeAgent::tick docstring for why.
  SafeTickResult tickSafe(const Eigen::Vector2d& current_pos, double current_yaw,
                          const Eigen::Vector2d& last_pos,
                          double stucking_distance, double soft_reach_distance);

  /// Forward to StrategicAgent::tick. Safe to call every FSM cycle — the
  /// agent rate-limits its LLM fires internally (currently a stub, see
  /// strategic_agent.{h,cpp}).
  bool tickStrategic();

  /// Non-owning views.
  SafeAgent*      safe()      { return safe_.get(); }
  StrategicAgent* strategic() { return strategic_.get(); }
  MemoryAgent*    memory()    { return memory_view_; }

 private:
  std::unique_ptr<SafeAgent>      safe_;
  std::unique_ptr<StrategicAgent> strategic_;
  MemoryAgent*                    memory_view_{nullptr};  ///< View; owner: ExplorationManager.
};

}  // namespace skillnav_planner
