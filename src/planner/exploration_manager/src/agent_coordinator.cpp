#include <exploration_manager/agent_coordinator.h>

#include <exploration_manager/exploration_manager.h>  // for memory_agent_ accessor

namespace skillnav_planner {

void AgentCoordinator::init(ros::NodeHandle& nh,
                            std::shared_ptr<FSMData> fd,
                            std::shared_ptr<ExplorationManager> em)
{
  // SafeAgent — owned here, lifetime tied to Coordinator.
  safe_ = std::make_unique<SafeAgent>();
  safe_->init(nh, fd, em);

  // StrategicAgent — also owned here. v0 is a stub but its plan is already
  // queryable via strategic()->currentPlan() for downstream instrumentation.
  strategic_ = std::make_unique<StrategicAgent>();
  strategic_->init(nh, fd, em);

  // MemoryAgent — still owned by ExplorationManager; we hold a raw view.
  // The accessor returns nullptr in build configs that skip MemoryAgent,
  // which is fine — callers must null-check memory() before use.
  if (em) {
    memory_view_ = em->getMemoryAgent();
  }
}

SafeTickResult AgentCoordinator::tickSafe(const Eigen::Vector2d& current_pos,
                                          double current_yaw,
                                          const Eigen::Vector2d& last_pos,
                                          double stucking_distance,
                                          double soft_reach_distance)
{
  return safe_->tick(current_pos, current_yaw, last_pos,
                     stucking_distance, soft_reach_distance);
}

bool AgentCoordinator::tickStrategic()
{
  return strategic_->tick();
}

}  // namespace skillnav_planner
