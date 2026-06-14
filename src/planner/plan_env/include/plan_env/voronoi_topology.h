#ifndef _VORONOI_TOPOLOGY_H_
#define _VORONOI_TOPOLOGY_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <boost/shared_ptr.hpp>
#include <visualization_msgs/MarkerArray.h>

namespace skillnav_planner {

class SDFMap2D;
class ValueMap;
class FrontierMap2D;
class MultiValueMapManager;

/**
 * @brief Dead zone status for VLM-confirmed obstacles
 */
enum DeadZoneStatus {
  DZ_NORMAL,              ///< Normal passable area
  DZ_TEMPORARILY_BLOCKED, ///< Temporary obstacle, will retry later
  DZ_HIGH_RISK,           ///< High risk area, reduced priority
  DZ_VLM_CONFIRMED_DEAD   ///< VLM confirmed permanent obstacle
};

/**
 * @brief Memory Agent data structure for topology nodes
 *
 * Stores VLM-generated semantic information anchored to Voronoi nodes.
 * Used for:
 * - Scene understanding and room type inference
 * - Object location memory for verification
 * - False positive tracking for semantic_multiplier adjustment
 */
struct NodeMemory {
  // === Core Semantic Information (VLM-generated) ===
  std::string scene_description;           ///< VLM scene description "A bathroom with white tiles"
  std::string inferred_room_type;          ///< Room type "bathroom", "bedroom", "kitchen", etc.
  std::vector<std::string> observed_objects;  ///< Objects seen ["toilet", "sink", "mirror"]

  // === Memory Metadata ===
  int visit_count = 0;                     ///< Number of times robot visited this node
  double first_visit_time = 0.0;           ///< ros::Time::now().toSec() of first visit
  double last_visit_time = 0.0;            ///< ros::Time::now().toSec() of last visit
  double last_vlm_update = 0.0;            ///< Last time VLM updated this memory
  float memory_confidence = 0.0f;          ///< Confidence in stored memory [0,1]

  // === State Flags ===
  bool description_generated = false;      ///< Whether VLM has generated description
  bool needs_update = false;               ///< Whether memory needs refresh (scene changed)
  bool verified = false;                   ///< Whether memory has been verified by VLM

  // === Verification History ===
  int verification_count = 0;              ///< Number of VLM verifications performed
  int false_positive_count = 0;            ///< Number of false positives detected

  NodeMemory() = default;

  /// Check if memory is stale (older than max_age_sec)
  bool isStale(double current_time, double max_age_sec = 300.0) const {
    if (!description_generated) return true;
    return (current_time - last_vlm_update) > max_age_sec;
  }

  /// Get a brief summary for logging
  std::string getSummary() const {
    if (!description_generated) return "[no memory]";
    std::string summary = inferred_room_type;
    if (!observed_objects.empty()) {
      summary += " (";
      for (size_t i = 0; i < std::min(size_t(3), observed_objects.size()); ++i) {
        if (i > 0) summary += ", ";
        summary += observed_objects[i];
      }
      summary += ")";
    }
    return summary;
  }

  /// Check if this memory contains a target object
  bool containsObject(const std::string& target) const {
    for (const auto& obj : observed_objects) {
      if (obj.find(target) != std::string::npos ||
          target.find(obj) != std::string::npos) {
        return true;
      }
    }
    return false;
  }
};

/**
 * @brief Topology node representing a junction point in the Voronoi diagram
 *
 * Extended for multi-agent control:
 * - Safe Agent: controls passability_multiplier via VLM confirmation
 * - Memory Agent: controls semantic_multiplier for FP tracking
 * - Exploration Agent: controls exploration_multiplier for phase-based weighting
 */
struct TopologyNode {
  // Architectural invariants for getEffectiveValue() — not tunable via ROS params,
  // they encode the contract that downstream code assumes about the value range.
  static constexpr double kMinEffective  = 0.01;  ///< Lower clamp on effective value
  static constexpr double kMaxEffective  = 1.0;   ///< Upper clamp on effective value
  static constexpr double kSemanticFloor = 0.1;   ///< Floor for semantic_multiplier under FP penalty

  int id;                          ///< Unique node identifier
  Eigen::Vector2d position;        ///< World position of the node

  // Value components for agent control (legacy)
  double base_value;               ///< Base value from ValueMap (read-only)
  double agent_additive;           ///< Agent-controllable additive modifier
  double agent_multiplier;         ///< Agent-controllable multiplicative modifier

  // Multi-agent control multipliers (new)
  double passability_multiplier;   ///< Safe Agent: obstacle/dead zone penalty [0.01, 1.0]
  double semantic_multiplier;      ///< Memory Agent: false positive penalty [0.1, 1.0]
  double exploration_multiplier;   ///< Exploration Agent: phase-based weight [0.5, 2.0]

  // Dead zone tracking (VLM-assisted)
  DeadZoneStatus dead_zone_status; ///< Current dead zone status
  double retry_time;               ///< Time (ros::Time::now().toSec()) to retry if temporarily blocked
  int escape_fail_count;           ///< Number of escape failures at this node

  // Node state
  enum NodeType { STABLE, FRONTIER_ADJACENT };
  NodeType type;                   ///< Whether node is near frontier (may change)
  bool is_valid;                   ///< Whether node is still valid
  int last_update_cycle;           ///< Last cycle this node was verified

  // Connectivity
  std::vector<int> neighbor_ids;   ///< IDs of connected topology nodes

  /// Degree of the node in the topology graph — equivalent to
  /// neighbor_ids.size() but kept as a separate cached int for readability
  /// and so future degree-bonus formulas have an obvious knob.
  int degree() const { return static_cast<int>(neighbor_ids.size()); }

  // ── C1 Hybrid Frontier-Topology anchoring ─────────────────────────────
  // Each cycle, every active Frontier2D is assigned to its nearest in-range
  // TopologyNode and that node's cells are aggregated into anchored_ig_sum.
  // Decision-time consumers will read these instead of iterating frontiers
  // directly; for the C-1 phase the data is computed and visualized but
  // not yet consumed.
  //
  // Both fields are RESET to empty / 0 at the start of every
  // VoronoiTopology::computeFrontierAnchors() pass — they represent the
  // current snapshot, not a running history.
  std::vector<int> anchored_frontier_ids;  ///< Frontier2D ids that map here
  double anchored_ig_sum;                  ///< Σ cells.size() of anchored frontiers

  // Memory Agent data (NEW)
  NodeMemory memory;               ///< VLM-generated semantic memory for this node

  TopologyNode() : id(-1), base_value(0.5), agent_additive(0.0),
                   agent_multiplier(1.0),
                   passability_multiplier(1.0), semantic_multiplier(1.0),
                   exploration_multiplier(1.0),
                   dead_zone_status(DZ_NORMAL), retry_time(0.0), escape_fail_count(0),
                   type(STABLE), is_valid(true), last_update_cycle(0),
                   anchored_ig_sum(0.0) {}

  /// Reset C1 anchoring state; called once at the start of every
  /// VoronoiTopology::computeFrontierAnchors() cycle.
  void clearFrontierAnchors() {
    anchored_frontier_ids.clear();
    anchored_ig_sum = 0.0;
  }

  /// Get effective value after all agent modifications
  double getEffectiveValue() const {
    double base = (base_value + agent_additive) * agent_multiplier;
    // Apply multi-agent multipliers
    double effective = base * passability_multiplier * semantic_multiplier * exploration_multiplier;
    return std::max(kMinEffective, std::min(kMaxEffective, effective));
  }

  /// Get effective value with phase-based IG/SR weighting
  double getEffectiveValue(double alpha_ig, double beta_sr, double ig_value, double sr_value) const {
    double base = alpha_ig * ig_value + beta_sr * sr_value;
    double effective = base * passability_multiplier * semantic_multiplier * exploration_multiplier;
    return std::max(kMinEffective, std::min(kMaxEffective, effective));
  }

  /// Check if this node should be retried (for temporarily blocked nodes)
  bool shouldRetry(double current_time) const {
    return dead_zone_status == DZ_TEMPORARILY_BLOCKED && current_time >= retry_time;
  }

  /// Reset penalties (called when node is successfully traversed)
  void resetPenalties() {
    passability_multiplier = 1.0;
    dead_zone_status = DZ_NORMAL;
    escape_fail_count = 0;
  }

  // === Memory Agent Interface ===

  /// Check if this node has VLM-generated memory
  bool hasMemory() const { return memory.description_generated; }

  /// Check if memory is stale and needs update
  bool isMemoryStale(double current_time, double max_age_sec = 300.0) const {
    return memory.isStale(current_time, max_age_sec);
  }

  /// Update visit count and time (called when robot passes through)
  void recordVisit(double current_time) {
    if (memory.visit_count == 0) {
      memory.first_visit_time = current_time;
    }
    memory.visit_count++;
    memory.last_visit_time = current_time;
  }

  /// Apply false positive penalty to semantic_multiplier
  void applyFalsePosivePenalty(float penalty_factor) {
    memory.false_positive_count++;
    // VLM decides the penalty factor, we just apply it
    semantic_multiplier *= penalty_factor;
    semantic_multiplier = std::max(kSemanticFloor, semantic_multiplier);
  }

  /// Check if this node's memory contains a target object
  bool containsTargetObject(const std::string& target) const {
    return memory.containsObject(target);
  }
};

/**
 * @brief Incremental Voronoi-based topology extraction for navigation
 *
 * This class maintains a GLOBAL Voronoi skeleton but only UPDATES the portion
 * within the robot's local sensing range. This provides:
 * - Full global coverage: See entire explored Voronoi diagram
 * - Fast updates: Only process local changes (~10ms per update)
 *
 * Algorithm based on Dynamic Brushfire (Lau et al. IROS 2010) and
 * G2VD Planner (arxiv:2201.12981)
 */
class VoronoiTopology {
public:
  VoronoiTopology() = default;
  ~VoronoiTopology();

  /**
   * @brief Initialize the topology extractor
   * @param nh ROS node handle
   * @param sdf_map Pointer to the SDF map
   */
  void init(ros::NodeHandle& nh, const std::shared_ptr<SDFMap2D>& sdf_map);

  /**
   * @brief Update topology based on map changes (incremental)
   * Only updates the local sensing region, preserves global structure
   */
  void updateTopology();

  /**
   * @brief Reset all topology data and clear visualization markers
   * Call this when starting a new episode to ensure clean state
   */
  void reset();

  /**
   * @brief Publish topology visualization to RViz
   */
  void publishVisualization();

  /**
   * @brief C1 Hybrid Frontier-Topology anchoring pass.
   *
   * For every Frontier2D currently active in the linked FrontierMap2D,
   * find the nearest TopologyNode within max_anchor_distance and store the
   * frontier's id on that node (anchored_frontier_ids) together with its
   * cell count (anchored_ig_sum). Frontiers further than
   * max_anchor_distance from every node remain *orphan* and are tracked
   * separately for downstream consumers (decision logic in C-2).
   *
   * @param max_anchor_distance  Meters — frontiers beyond this distance
   *        from every node remain unanchored. Defaults to
   *        frontier_distance_ * 1.5 (the same threshold the visualization
   *        layer was already using internally).
   *
   * Resets each node's anchored fields at the start of the call, so
   * snapshots from previous cycles do not bleed into the current state.
   * No-op (and returns 0) if frontier_map_ is null.
   *
   * @return Number of frontiers that were successfully anchored
   *         (0 to total_frontier_count). Useful for ROS_DEBUG telemetry.
   */
  int computeFrontierAnchors(double max_anchor_distance = -1.0);

  /// Centroids of frontiers that were too far from every node during the
  /// most recent computeFrontierAnchors() call. Decision logic must still
  /// consider these (e.g. C-1 falls back to legacy per-frontier scoring
  /// for orphans).
  const std::vector<Eigen::Vector2d>& getOrphanFrontiers() const {
    return orphan_frontiers_;
  }

  // Accessors
  const std::vector<TopologyNode>& getNodes() const { return nodes_; }
  std::vector<TopologyNode>& getNodes() { return nodes_; }

  // Agent control interface
  void setNodeAdditive(int node_id, double additive);
  void setNodeMultiplier(int node_id, double multiplier);
  TopologyNode* getNearestNode(const Eigen::Vector2d& pos);

  // Memory Agent interface
  /**
   * @brief Get nearby nodes with memory within search radius
   * @param pos Current position
   * @param radius Search radius in meters (default 1.5m)
   * @param max_nodes Maximum nodes to return (default 3)
   * @return Vector of pairs: (node_id, distance)
   */
  std::vector<std::pair<int, double>> getNearbyNodesWithMemory(
      const Eigen::Vector2d& pos, double radius = 1.5, int max_nodes = 3) const;

  /**
   * @brief Get node by ID (const version for reading)
   */
  const TopologyNode* getNodeById(int node_id) const;

  /**
   * @brief Get node by ID (mutable version for updating)
   */
  TopologyNode* getNodeById(int node_id);

  /// Set frontier map for node-frontier link visualization
  void setFrontierMap(const std::shared_ptr<FrontierMap2D>& frontier_map) {
    frontier_map_ = frontier_map;
  }

  /// Attach MultiValueMapManager so node base_value is sampled from the fused IG+SR map.
  /// Falls back to the legacy single value_map_ when null.
  void setMultiValueMapManager(MultiValueMapManager* mvm) { mvm_ = mvm; }

private:
  // Core incremental algorithm
  void updateLocalVoronoiPoints();      ///< Update Voronoi flag in local region
  void mergeLocalNodes();               ///< Merge new local nodes into global set
  void pruneInvalidNodes();             ///< Remove nodes in newly occupied areas
  void buildNodeConnections();          ///< Build/update node connectivity
  void updateNodeValues();              ///< Update values from ValueMap

  // Helper functions
  bool isVoronoiPoint(const Eigen::Vector2i& idx);
  bool isJunctionPoint(const Eigen::Vector2i& idx);
  int countVoronoiNeighbors(const Eigen::Vector2i& idx);
  int countVoronoiBranches(const Eigen::Vector2i& idx);  ///< Count distinct Voronoi branches
  std::vector<Eigen::Vector2i> getNeighbors(const Eigen::Vector2i& idx);
  double computeGradientVariance(const Eigen::Vector2i& idx);
  int64_t posToKey(const Eigen::Vector2d& pos);  ///< Spatial hash for node lookup

  // Data - GLOBAL storage
  // Note: This shared_ptr creates a reference to SDFMap2D, but we do NOT reset it
  // in destructor - SDFMap2D is responsible for destroying us, so we just let
  // this reference become invalid naturally when SDFMap2D is destroyed
  std::shared_ptr<SDFMap2D> sdf_map_;
  std::shared_ptr<FrontierMap2D> frontier_map_;  ///< Frontier map for visualization
  MultiValueMapManager* mvm_ = nullptr;     ///< Optional fused value map source
  std::atomic<bool> is_destroying_{false};  ///< Flag to prevent callback during destruction
  // Token tracked by the ROS timer (weak_ptr semantics). Reset in destructor so
  // any callback still queued in CallbackQueue after we die is safely skipped
  // instead of dispatching on freed memory. boost::shared_ptr to match
  // ros::TimerOptions::tracked_object's expected type.
  boost::shared_ptr<int> alive_token_;
  std::vector<TopologyNode> nodes_;              ///< Global node storage
  /// C1 anchoring: centroids of frontiers that fell outside the anchor
  /// radius of every node during the last computeFrontierAnchors() call.
  /// Reset each call. Decision logic falls back to legacy per-frontier
  /// scoring for these.
  std::vector<Eigen::Vector2d> orphan_frontiers_;
  std::vector<char> voronoi_flag_;               ///< Global Voronoi point flags
  std::unordered_map<int64_t, int> spatial_hash_;///< Fast spatial lookup for nodes
  int next_node_id_;
  int update_cycle_;                             ///< Current update cycle number
  int last_published_node_count_ = 0;            ///< For per-tick DELETE of removed decision_node IDs

  // Local update region tracking
  Eigen::Vector2d last_update_min_;
  Eigen::Vector2d last_update_max_;

  // Parameters
  double min_node_distance_;        ///< Minimum distance between nodes
  double min_obstacle_distance_;    ///< Minimum distance from obstacles (> robot radius)
  double gradient_threshold_;       ///< Threshold for gradient discontinuity
  double frontier_distance_;        ///< Distance to consider node as frontier-adjacent
  int node_lifetime_cycles_;        ///< Cycles before removing unvisited nodes
  bool show_edges_;                 ///< Whether to show edges between nodes

  // ROS
  ros::NodeHandle nh_;
  ros::Publisher topology_pub_;
  ros::Publisher skeleton_pub_;
  ros::Publisher node_frontier_link_pub_;  ///< Visualize node-frontier connections
  ros::Timer update_timer_;
  std::string frame_id_;
  bool show_node_frontier_links_;          ///< Whether to show node-frontier connections

  void updateCallback(const ros::TimerEvent& event);

public:
  typedef std::shared_ptr<VoronoiTopology> Ptr;
};

}  // namespace skillnav_planner

#endif  // _VORONOI_TOPOLOGY_H_
