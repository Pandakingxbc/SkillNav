/**
 * @file voronoi_topology.cpp
 * @brief Sparse topological graph extraction for navigation
 *
 * Inspired by recent top-venue papers (CVPR, IROS, CoRL):
 * - ETPNav (ICRA 2023): Waypoints at decision points only
 * - VLN-VER (CVPR 2024): Nodes at visited viewpoints and key junctions
 * - TopoNav: Sparse graphs for efficient planning
 *
 * Key principle: Nodes represent DECISION POINTS, not dense path coverage.
 * A decision point is where the robot must choose between 3+ directions.
 */

#include <plan_env/voronoi_topology.h>
#include <ros/timer_options.h>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>
#include <plan_env/frontier_map2d.h>
#include <plan_env/multi_valuemap_manager.h>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace skillnav_planner {

VoronoiTopology::~VoronoiTopology() {
  ROS_WARN("[VoronoiTopology] ===== DESTRUCTOR CALLED ===== clearing %zu nodes", nodes_.size());

  // Set destroying flag FIRST to prevent any callbacks from accessing members
  is_destroying_.store(true);

  // CRITICAL: Stop the timer and invalidate tracked_object so any queued
  // callback in CallbackQueue is skipped (its boost weak_ptr fails to lock).
  if (update_timer_) {
    update_timer_.stop();
  }
  alive_token_.reset();
  // Give any in-progress callback time to check the flag and exit.
  ros::Duration(0.05).sleep();

  // Clear visualization markers (only if ROS is still running)
  if (ros::ok()) {
    auto createDeleteMarker = [&](const std::string& ns) {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id_.empty() ? "world" : frame_id_;
      m.header.stamp = ros::Time::now();
      m.ns = ns;
      m.id = 0;
      m.action = visualization_msgs::Marker::DELETEALL;
      return m;
    };

    visualization_msgs::MarkerArray clear_all;
    clear_all.markers.push_back(createDeleteMarker("decision_nodes"));
    clear_all.markers.push_back(createDeleteMarker("topology_edges"));
    clear_all.markers.push_back(createDeleteMarker("voronoi_skeleton"));
    clear_all.markers.push_back(createDeleteMarker("node_frontier_links"));
    clear_all.markers.push_back(createDeleteMarker("frontier_centers"));

    // Publish to all topics (check if publishers are valid)
    if (topology_pub_ && topology_pub_.getNumSubscribers() >= 0)
      topology_pub_.publish(clear_all);
    if (skeleton_pub_ && skeleton_pub_.getNumSubscribers() >= 0)
      skeleton_pub_.publish(clear_all);
    if (node_frontier_link_pub_ && node_frontier_link_pub_.getNumSubscribers() >= 0)
      node_frontier_link_pub_.publish(clear_all);

    ros::Duration(0.05).sleep();
  }

  // Clear data structures
  nodes_.clear();
  voronoi_flag_.clear();
  spatial_hash_.clear();

  // Clear frontier_map_ shared_ptr
  // NOTE: Do NOT reset sdf_map_ here - it owns us, so resetting it would
  // cause issues. Just let it become invalid naturally.
  frontier_map_.reset();

  ROS_WARN("[VoronoiTopology] ===== DESTRUCTOR COMPLETE =====");
}

void VoronoiTopology::init(ros::NodeHandle& nh,
                           const std::shared_ptr<SDFMap2D>& sdf_map) {
  nh_ = nh;
  sdf_map_ = sdf_map;  // Store shared_ptr (creates reference, but we don't reset in dtor)
  is_destroying_.store(false);  // Reset flag on init
  next_node_id_ = 0;
  update_cycle_ = 0;

  // Parameters based on robot step size (0.25m) and robot radius (0.18m)
  nh_.param("voronoi_topology/min_node_distance", min_node_distance_, 1.5);  // ~6x step size
  nh_.param("voronoi_topology/gradient_threshold", gradient_threshold_, 0.25);
  nh_.param("voronoi_topology/frontier_distance", frontier_distance_, 2.0);
  nh_.param("voronoi_topology/frame_id", frame_id_, std::string("world"));
  nh_.param("voronoi_topology/node_lifetime_cycles", node_lifetime_cycles_, 100);
  nh_.param("voronoi_topology/show_edges", show_edges_, false);
  nh_.param("voronoi_topology/min_obstacle_distance", min_obstacle_distance_, 0.3);  // > robot radius

  // Initialize global voronoi flag array
  int voxel_num = sdf_map_->getVoxelNum();
  voronoi_flag_.resize(voxel_num, 0);

  // Setup publishers
  topology_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/voronoi_topology/nodes", 10);
  skeleton_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/voronoi_topology/skeleton", 10);
  node_frontier_link_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/voronoi_topology/node_frontier_links", 10);

  // Burst DELETE markers for a wide ID range so any leftover decision_nodes
  // from a previous episode get wiped before NEW VT starts publishing fresh
  // IDs (otherwise RViz holds them until lifetime expires). RViz processes
  // explicit DELETE more reliably than DELETEALL on its MarkerArray Display.
  {
    visualization_msgs::MarkerArray purge;
    for (int id = 0; id < 200; ++id) {
      visualization_msgs::Marker d;
      d.header.frame_id = frame_id_;
      d.header.stamp = ros::Time::now();
      d.ns = "decision_nodes";
      d.id = id;
      d.action = visualization_msgs::Marker::DELETE;
      purge.markers.push_back(d);
    }
    topology_pub_.publish(purge);
  }

  // Parameter for node-frontier link visualization
  nh_.param("voronoi_topology/show_node_frontier_links", show_node_frontier_links_, true);

  // Update at 1Hz - sparse topology doesn't need frequent updates.
  // Use TimerOptions::tracked_object so any callback still in CallbackQueue
  // after we are destroyed is skipped (boost weak_ptr fails to lock).
  alive_token_ = boost::make_shared<int>(1);
  ros::TimerOptions topts;
  topts.period = ros::Duration(1.0);
  topts.callback = boost::bind(&VoronoiTopology::updateCallback, this, boost::placeholders::_1);
  topts.tracked_object = alive_token_;
  update_timer_ = nh_.createTimer(topts);

  ROS_INFO("[VoronoiTopology] Sparse topology: min_dist=%.1fm, min_obs_dist=%.2fm",
           min_node_distance_, min_obstacle_distance_);
}

void VoronoiTopology::updateCallback(const ros::TimerEvent& event) {
  // Check if we're being destroyed - exit immediately if so
  if (is_destroying_.load()) {
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();

  updateTopology();
  // C1 anchoring pass: assign each frontier to its nearest node and
  // aggregate IG cells onto the node. Cheap (O(frontiers × nodes)) and
  // self-contained — the result lands in TopologyNode::anchored_*
  // fields and orphan_frontiers_, ready for the C-2 decision layer to
  // consume in a follow-up commit.
  computeFrontierAnchors();
  publishVisualization();

  auto end = std::chrono::high_resolution_clock::now();
  double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

  ROS_INFO_THROTTLE(10.0, "[VoronoiTopology] %zu decision nodes, %.1f ms",
                    nodes_.size(), duration_ms);
}

void VoronoiTopology::reset() {
  ROS_INFO("[VoronoiTopology] Resetting topology for new episode...");

  // Clear all data structures
  nodes_.clear();
  spatial_hash_.clear();
  std::fill(voronoi_flag_.begin(), voronoi_flag_.end(), 0);
  next_node_id_ = 0;
  update_cycle_ = 0;
  last_published_node_count_ = 0;

  // Publish DELETEALL markers to clear RViz visualization
  auto createDeleteMarker = [&](const std::string& ns) {
    visualization_msgs::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.id = 0;
    m.action = visualization_msgs::Marker::DELETEALL;
    return m;
  };

  // Clear node markers
  visualization_msgs::MarkerArray clear_nodes;
  clear_nodes.markers.push_back(createDeleteMarker("decision_nodes"));
  topology_pub_.publish(clear_nodes);

  // Clear skeleton markers
  visualization_msgs::MarkerArray clear_skeleton;
  clear_skeleton.markers.push_back(createDeleteMarker("topology_edges"));
  clear_skeleton.markers.push_back(createDeleteMarker("voronoi_skeleton"));
  skeleton_pub_.publish(clear_skeleton);

  // Clear node-frontier link markers
  visualization_msgs::MarkerArray clear_links;
  clear_links.markers.push_back(createDeleteMarker("node_frontier_links"));
  clear_links.markers.push_back(createDeleteMarker("frontier_centers"));
  node_frontier_link_pub_.publish(clear_links);

  ROS_INFO("[VoronoiTopology] Reset complete - all markers cleared");
}

void VoronoiTopology::updateTopology() {
  if (!sdf_map_ || is_destroying_.load()) return;

  update_cycle_++;

  // Step 1: Update Voronoi skeleton in local region
  updateLocalVoronoiPoints();

  // Step 2: Extract ONLY true decision points (3+ branches)
  mergeLocalNodes();

  // Step 3: Remove invalid nodes
  pruneInvalidNodes();

  // Step 4: Build sparse connections
  buildNodeConnections();

  // Step 5: Update values
  updateNodeValues();
}

void VoronoiTopology::updateLocalVoronoiPoints() {
  Eigen::Vector2d local_min, local_max;
  sdf_map_->getLocalUpdatedBox(local_min, local_max);

  double region_size = (local_max - local_min).norm();
  if (region_size < 0.1) return;

  last_update_min_ = local_min;
  last_update_max_ = local_max;

  double resolution = sdf_map_->getResolution();

  // Clear local region
  for (double x = local_min.x(); x < local_max.x(); x += resolution) {
    for (double y = local_min.y(); y < local_max.y(); y += resolution) {
      Eigen::Vector2d pos(x, y);
      Eigen::Vector2i idx;
      sdf_map_->posToIndex(pos, idx);
      int adr = sdf_map_->toAddress(idx);
      if (adr >= 0 && static_cast<size_t>(adr) < voronoi_flag_.size()) {
        voronoi_flag_[adr] = 0;
      }
    }
  }

  // Compute Voronoi points
  for (double x = local_min.x() + resolution; x < local_max.x() - resolution; x += resolution) {
    for (double y = local_min.y() + resolution; y < local_max.y() - resolution; y += resolution) {
      Eigen::Vector2d pos(x, y);
      Eigen::Vector2i idx;
      sdf_map_->posToIndex(pos, idx);

      if (sdf_map_->getOccupancy(idx) != SDFMap2D::FREE) continue;

      if (isVoronoiPoint(idx)) {
        int adr = sdf_map_->toAddress(idx);
        if (adr >= 0 && static_cast<size_t>(adr) < voronoi_flag_.size()) {
          voronoi_flag_[adr] = 1;
        }
      }
    }
  }
}

void VoronoiTopology::mergeLocalNodes() {
  if ((last_update_max_ - last_update_min_).norm() < 0.1) return;

  double resolution = sdf_map_->getResolution();

  // Find junction points: places where Voronoi skeleton branches
  std::vector<std::pair<Eigen::Vector2d, int>> decision_points;

  // First count total voronoi points for debugging
  int voronoi_count = 0;
  for (double x = last_update_min_.x() + resolution; x < last_update_max_.x() - resolution; x += resolution) {
    for (double y = last_update_min_.y() + resolution; y < last_update_max_.y() - resolution; y += resolution) {
      Eigen::Vector2d pos(x, y);
      Eigen::Vector2i idx;
      sdf_map_->posToIndex(pos, idx);

      int adr = sdf_map_->toAddress(idx);
      if (adr < 0 || static_cast<size_t>(adr) >= voronoi_flag_.size()) continue;
      if (!voronoi_flag_[adr]) continue;

      voronoi_count++;

      // Count Voronoi neighbors (simpler than branch counting)
      int neighbors = countVoronoiNeighbors(idx);

      // Junction = point with 3+ Voronoi neighbors OR isolated point with 1 neighbor (endpoint)
      // Corridor = point with exactly 2 neighbors (skip these)
      if (neighbors >= 3 || neighbors == 1) {
        decision_points.push_back({pos, neighbors});
      }
    }
  }

  ROS_INFO_THROTTLE(5.0, "[VoronoiTopology] Local region: %d Voronoi pts, %zu candidates",
                    voronoi_count, decision_points.size());

  // Sort by branch count (prioritize major junctions)
  std::sort(decision_points.begin(), decision_points.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Add nodes with minimum distance constraints
  int rejected_obstacle = 0, rejected_distance = 0, accepted = 0;
  for (const auto& dp : decision_points) {
    const Eigen::Vector2d& pos = dp.first;

    // Check distance to obstacles - must be > robot radius
    Eigen::Vector2d grad;
    double dist_to_obstacle = sdf_map_->getDistWithGrad(pos, grad);
    if (dist_to_obstacle < min_obstacle_distance_) {
      rejected_obstacle++;
      continue;  // Too close to obstacle, skip
    }

    // Check distance from ALL existing nodes
    bool too_close = false;
    for (const auto& node : nodes_) {
      if ((node.position - pos).norm() < min_node_distance_) {
        too_close = true;
        break;
      }
    }

    if (too_close) {
      rejected_distance++;
      continue;
    }
    accepted++;

    // Create new decision node
    TopologyNode node;
    node.id = next_node_id_++;
    node.position = pos;
    node.is_valid = true;
    node.last_update_cycle = update_cycle_;

    // Check if near frontier
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(pos, idx);
    bool near_frontier = false;
    for (int dx = -5; dx <= 5 && !near_frontier; dx++) {
      for (int dy = -5; dy <= 5 && !near_frontier; dy++) {
        Eigen::Vector2i check_idx = idx + Eigen::Vector2i(dx, dy);
        if (sdf_map_->isInMap(check_idx) &&
            sdf_map_->getOccupancy(check_idx) == SDFMap2D::UNKNOWN) {
          near_frontier = true;
        }
      }
    }
    node.type = near_frontier ? TopologyNode::FRONTIER_ADJACENT : TopologyNode::STABLE;

    nodes_.push_back(node);
    spatial_hash_[posToKey(pos)] = node.id;
  }

  // Debug: show rejection reasons
  if (rejected_obstacle > 0 || rejected_distance > 0) {
    ROS_INFO_THROTTLE(5.0, "[VoronoiTopology] Candidates: %zu, rejected(obs): %d, rejected(dist): %d, accepted: %d, total_nodes: %zu",
                      decision_points.size(), rejected_obstacle, rejected_distance, accepted, nodes_.size());
  }

  // Update cycle for existing nodes in local region
  for (auto& node : nodes_) {
    if (node.position.x() >= last_update_min_.x() &&
        node.position.x() <= last_update_max_.x() &&
        node.position.y() >= last_update_min_.y() &&
        node.position.y() <= last_update_max_.y()) {
      node.last_update_cycle = update_cycle_;
    }
  }
}

int VoronoiTopology::countVoronoiBranches(const Eigen::Vector2i& idx) {
  // Count distinct branches by checking transitions in 8-connectivity circle
  int branches = 0;
  bool prev_is_voronoi = false;

  const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

  // Check if first neighbor is Voronoi (for wrap-around)
  Eigen::Vector2i first_nbr = idx + Eigen::Vector2i(dx[0], dy[0]);
  bool first_is_voronoi = false;
  if (sdf_map_->isInMap(first_nbr)) {
    int adr = sdf_map_->toAddress(first_nbr);
    if (adr >= 0 && static_cast<size_t>(adr) < voronoi_flag_.size()) {
      first_is_voronoi = voronoi_flag_[adr];
    }
  }

  // Count transitions from non-Voronoi to Voronoi
  for (int i = 0; i < 8; i++) {
    Eigen::Vector2i nbr = idx + Eigen::Vector2i(dx[i], dy[i]);
    bool is_voronoi = false;

    if (sdf_map_->isInMap(nbr)) {
      int adr = sdf_map_->toAddress(nbr);
      if (adr >= 0 && static_cast<size_t>(adr) < voronoi_flag_.size()) {
        is_voronoi = voronoi_flag_[adr];
      }
    }

    if (is_voronoi && !prev_is_voronoi) {
      branches++;
    }
    prev_is_voronoi = is_voronoi;
  }

  // Handle wrap-around: if last and first are both Voronoi, they're connected
  if (prev_is_voronoi && first_is_voronoi && branches > 0) {
    branches--;
  }

  return branches;
}

void VoronoiTopology::pruneInvalidNodes() {
  auto it = nodes_.begin();
  while (it != nodes_.end()) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(it->position, idx);

    if (sdf_map_->getOccupancy(idx) != SDFMap2D::FREE) {
      spatial_hash_.erase(posToKey(it->position));
      it = nodes_.erase(it);
    } else {
      ++it;
    }
  }
}

int64_t VoronoiTopology::posToKey(const Eigen::Vector2d& pos) {
  int32_t ix = static_cast<int32_t>(pos.x() * 10.0);
  int32_t iy = static_cast<int32_t>(pos.y() * 10.0);
  return (static_cast<int64_t>(ix) << 32) | static_cast<int64_t>(iy);
}

bool VoronoiTopology::isVoronoiPoint(const Eigen::Vector2i& idx) {
  double variance = computeGradientVariance(idx);
  return variance > gradient_threshold_;
}

double VoronoiTopology::computeGradientVariance(const Eigen::Vector2i& idx) {
  std::vector<Eigen::Vector2d> gradients;
  gradients.reserve(8);

  auto neighbors = getNeighbors(idx);

  for (const auto& nbr : neighbors) {
    if (!sdf_map_->isInMap(nbr)) continue;
    if (sdf_map_->getOccupancy(nbr) != SDFMap2D::FREE) continue;

    Eigen::Vector2d nbr_pos, grad;
    sdf_map_->indexToPos(nbr, nbr_pos);
    sdf_map_->getDistWithGrad(nbr_pos, grad);

    if (grad.norm() > 1e-6) {
      gradients.push_back(grad.normalized());
    }
  }

  if (gradients.size() < 3) return 0.0;

  double variance = 0.0;
  int count = 0;
  for (size_t i = 0; i < gradients.size(); i++) {
    for (size_t j = i + 1; j < gradients.size(); j++) {
      double dot = gradients[i].dot(gradients[j]);
      variance += (1.0 - dot) / 2.0;
      count++;
    }
  }

  return count > 0 ? variance / count : 0.0;
}

bool VoronoiTopology::isJunctionPoint(const Eigen::Vector2i& idx) {
  return countVoronoiBranches(idx) >= 3;
}

int VoronoiTopology::countVoronoiNeighbors(const Eigen::Vector2i& idx) {
  int count = 0;
  auto neighbors = getNeighbors(idx);
  for (const auto& nbr : neighbors) {
    if (!sdf_map_->isInMap(nbr)) continue;
    int adr = sdf_map_->toAddress(nbr);
    if (adr >= 0 && static_cast<size_t>(adr) < voronoi_flag_.size() && voronoi_flag_[adr]) {
      count++;
    }
  }
  return count;
}

std::vector<Eigen::Vector2i> VoronoiTopology::getNeighbors(const Eigen::Vector2i& idx) {
  std::vector<Eigen::Vector2i> neighbors;
  neighbors.reserve(8);
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0) continue;
      neighbors.push_back(idx + Eigen::Vector2i(dx, dy));
    }
  }
  return neighbors;
}

void VoronoiTopology::buildNodeConnections() {
  // Sparse connectivity: only connect nearby decision nodes
  double max_connect_distance = min_node_distance_ * 2.0;

  for (auto& node : nodes_) {
    node.neighbor_ids.clear();
    for (const auto& other : nodes_) {
      if (node.id == other.id) continue;
      double dist = (node.position - other.position).norm();
      if (dist < max_connect_distance) {
        node.neighbor_ids.push_back(other.id);
      }
    }
  }
}

void VoronoiTopology::updateNodeValues() {
  if (!sdf_map_ || is_destroying_.load()) return;

  // Prefer fused IG+SR map when MVMM is attached; fall back to legacy value_map_.
  if (mvm_ != nullptr) {
    for (auto& node : nodes_) {
      Eigen::Vector2i idx;
      sdf_map_->posToIndex(node.position, idx);
      node.base_value = mvm_->getCombinedValueAtGrid(idx);
    }
    return;
  }

  if (!sdf_map_->value_map_) return;
  for (auto& node : nodes_) {
    Eigen::Vector2i idx;
    sdf_map_->posToIndex(node.position, idx);
    node.base_value = sdf_map_->value_map_->getValue(idx);
  }
}

int VoronoiTopology::computeFrontierAnchors(double max_anchor_distance) {
  // Reset per-cycle state.
  for (auto& node : nodes_) node.clearFrontierAnchors();
  orphan_frontiers_.clear();

  if (!frontier_map_ || nodes_.empty()) return 0;

  std::vector<std::vector<Eigen::Vector2d>> frontier_clusters;
  std::vector<Eigen::Vector2d> frontier_averages;
  frontier_map_->getFrontiers(frontier_clusters, frontier_averages);
  if (frontier_averages.empty()) return 0;

  const double radius = (max_anchor_distance > 0.0)
                           ? max_anchor_distance
                           : frontier_distance_ * 1.5;

  // FrontierMap2D currently does not expose a stable per-cluster id (the
  // public interface only hands back averages + cell arrays). For C-1
  // we therefore use the frontier's INDEX in the frontier_averages array
  // as its anchored id — this is stable for the duration of one tick and
  // round-trips with frontier_clusters[idx] for cell-count lookup, which
  // is all C-2 decision logic needs.
  int anchored = 0;
  for (size_t fi = 0; fi < frontier_averages.size(); ++fi) {
    const auto& fpos = frontier_averages[fi];

    TopologyNode* nearest = nullptr;
    double min_dist = radius;
    for (auto& node : nodes_) {
      if (!node.is_valid) continue;
      const double d = (node.position - fpos).norm();
      if (d < min_dist) {
        min_dist = d;
        nearest = &node;
      }
    }

    if (nearest != nullptr) {
      nearest->anchored_frontier_ids.push_back(static_cast<int>(fi));
      // IG proxy: number of cells in the frontier cluster. Per-cell weight
      // is normalised in C-2; here we only need the raw count.
      const double cell_count =
          (fi < frontier_clusters.size())
              ? static_cast<double>(frontier_clusters[fi].size())
              : 0.0;
      nearest->anchored_ig_sum += cell_count;
      ++anchored;
    } else {
      orphan_frontiers_.push_back(fpos);
    }
  }

  ROS_DEBUG_THROTTLE(5.0,
      "[VT/Anchor] frontiers=%zu, anchored=%d, orphan=%zu, nodes=%zu",
      frontier_averages.size(), anchored, orphan_frontiers_.size(),
      nodes_.size());

  return anchored;
}

void VoronoiTopology::publishVisualization() {
  if (is_destroying_.load()) return;

  if (topology_pub_.getNumSubscribers() == 0 &&
      skeleton_pub_.getNumSubscribers() == 0) {
    return;
  }

  visualization_msgs::MarkerArray node_markers;
  visualization_msgs::MarkerArray skeleton_markers;

  // Per-tick DELETEALL on each namespace before re-adding markers. This is
  // the ApexNav-style approach: simpler than per-ID DELETE tracking and
  // matches how ApexNav handles its frontier visualisation by always
  // re-publishing the entire set each tick.
  auto createClearMarker = [&](const std::string& ns) {
    visualization_msgs::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.id = 0;
    m.action = visualization_msgs::Marker::DELETEALL;
    return m;
  };
  node_markers.markers.push_back(createClearMarker("decision_nodes"));
  skeleton_markers.markers.push_back(createClearMarker("topology_edges"));
  skeleton_markers.markers.push_back(createClearMarker("voronoi_skeleton"));

  // Publish decision nodes as larger, prominent spheres.
  // lifetime=2s safety net: if this VoronoiTopology stops publishing (process
  // SIGKILLed at episode reset, or RViz misses the DELETE burst due to
  // pub-sub discovery latency), RViz auto-expires the markers ~2s after the
  // last republish. Republish period is 1s so a single missed tick is OK.
  for (const auto& node : nodes_) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "decision_nodes";
    marker.id = node.id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(2.0);
    marker.pose.position.x = node.position.x();
    marker.pose.position.y = node.position.y();
    marker.pose.position.z = 0.4;
    marker.pose.orientation.w = 1.0;

    // Prominent size for decision points
    marker.scale.x = marker.scale.y = marker.scale.z = 0.35;

    // Color: cyan for frontier-adjacent, orange for stable
    if (node.type == TopologyNode::FRONTIER_ADJACENT) {
      marker.color.r = 0.0;
      marker.color.g = 0.8;
      marker.color.b = 1.0;
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.6;
      marker.color.b = 0.0;
    }
    marker.color.a = 0.9;

    node_markers.markers.push_back(marker);
  }

  // Optionally show edges
  if (show_edges_ && nodes_.size() > 1) {
    visualization_msgs::Marker edge_marker;
    edge_marker.header.frame_id = frame_id_;
    edge_marker.header.stamp = ros::Time::now();
    edge_marker.ns = "topology_edges";
    edge_marker.id = 0;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;
    edge_marker.action = visualization_msgs::Marker::ADD;
    edge_marker.lifetime = ros::Duration(2.0);
    edge_marker.scale.x = 0.06;
    edge_marker.color.r = 0.5;
    edge_marker.color.g = 0.5;
    edge_marker.color.b = 0.5;
    edge_marker.color.a = 0.5;
    edge_marker.pose.orientation.w = 1.0;

    std::unordered_map<int, const TopologyNode*> id_to_node;
    for (const auto& node : nodes_) {
      id_to_node[node.id] = &node;
    }

    for (const auto& node : nodes_) {
      for (int neighbor_id : node.neighbor_ids) {
        if (node.id >= neighbor_id) continue;
        auto it = id_to_node.find(neighbor_id);
        if (it == id_to_node.end()) continue;

        geometry_msgs::Point p1, p2;
        p1.x = node.position.x();
        p1.y = node.position.y();
        p1.z = 0.35;
        p2.x = it->second->position.x();
        p2.y = it->second->position.y();
        p2.z = 0.35;

        edge_marker.points.push_back(p1);
        edge_marker.points.push_back(p2);
      }
    }

    // Only push if we have at least one edge — pushing an empty LINE_LIST
    // marker triggers "Topic Status: Error" in RViz. Lifetime=2s handles
    // cleanup when edges are gone.
    if (!edge_marker.points.empty()) {
      skeleton_markers.markers.push_back(edge_marker);
    }
  }

  // Publish lightweight skeleton (very sparse sampling)
  if (skeleton_pub_.getNumSubscribers() > 0) {
    visualization_msgs::Marker skeleton_marker;
    skeleton_marker.header.frame_id = frame_id_;
    skeleton_marker.header.stamp = ros::Time::now();
    skeleton_marker.ns = "voronoi_skeleton";
    skeleton_marker.id = 0;
    skeleton_marker.type = visualization_msgs::Marker::POINTS;
    skeleton_marker.action = visualization_msgs::Marker::ADD;
    skeleton_marker.lifetime = ros::Duration(2.0);
    skeleton_marker.scale.x = skeleton_marker.scale.y = 0.02;
    skeleton_marker.color.r = 0.4;
    skeleton_marker.color.g = 0.4;
    skeleton_marker.color.b = 0.4;
    skeleton_marker.color.a = 0.2;
    skeleton_marker.pose.orientation.w = 1.0;

    // Very sparse sampling (every 8th point)
    int sample_rate = 8;
    int count = 0;
    for (size_t i = 0; i < voronoi_flag_.size(); i++) {
      if (voronoi_flag_[i]) {
        if (count % sample_rate == 0) {
          Eigen::Vector2i idx = sdf_map_->addressToIdx(i);
          Eigen::Vector2d pos;
          sdf_map_->indexToPos(idx, pos);

          geometry_msgs::Point p;
          p.x = pos.x();
          p.y = pos.y();
          p.z = 0.05;
          skeleton_marker.points.push_back(p);
        }
        count++;
      }
    }

    // Only push if we have skeleton points. Lifetime handles cleanup.
    if (!skeleton_marker.points.empty()) {
      skeleton_markers.markers.push_back(skeleton_marker);
    }
  }

  // Skip empty MarkerArray publish — RViz marks the Display as "Topic Status:
  // Error" if it receives an empty array. Lifetime=2s on previously published
  // markers handles cleanup; we don't need to send anything when there's
  // nothing to show.
  if (!node_markers.markers.empty()) topology_pub_.publish(node_markers);
  if (!skeleton_markers.markers.empty()) skeleton_pub_.publish(skeleton_markers);

  // Publish node-frontier links visualization
  // Always publish if enabled (don't check subscribers - RViz may add later)
  ROS_INFO_THROTTLE(10.0, "[VoronoiTopology] Link vis: show=%d, frontier_map=%s",
                    show_node_frontier_links_, frontier_map_ ? "SET" : "NULL");
  if (show_node_frontier_links_ && frontier_map_) {
    visualization_msgs::MarkerArray link_markers;
    // Skip per-tick DELETEALL (was causing visible flicker). The LINE_LIST
    // and SPHERE_LIST markers below have fixed IDs and are published every
    // tick with ADD, which RViz treats as overwrite. Episode-level
    // DELETEALL is still emitted from reset()/destructor.
    //
    // Source of truth: TopologyNode::anchored_frontier_ids already filled
    // by the preceding computeFrontierAnchors() call. We grab the frontier
    // centroid array once for coordinate lookup.
    std::vector<std::vector<Eigen::Vector2d>> frontier_clusters;
    std::vector<Eigen::Vector2d> frontier_averages;
    frontier_map_->getFrontiers(frontier_clusters, frontier_averages);

    if (!frontier_averages.empty()) {
      visualization_msgs::Marker link_marker;
      link_marker.header.frame_id = frame_id_;
      link_marker.header.stamp = ros::Time::now();
      link_marker.ns = "node_frontier_links";
      link_marker.id = 0;
      link_marker.type = visualization_msgs::Marker::LINE_LIST;
      link_marker.action = visualization_msgs::Marker::ADD;
      link_marker.lifetime = ros::Duration(2.0);
      link_marker.scale.x = 0.03;
      link_marker.pose.orientation.w = 1.0;

      visualization_msgs::Marker frontier_center_marker;
      frontier_center_marker.header.frame_id = frame_id_;
      frontier_center_marker.header.stamp = ros::Time::now();
      frontier_center_marker.ns = "frontier_centers";
      frontier_center_marker.id = 1;
      frontier_center_marker.type = visualization_msgs::Marker::SPHERE_LIST;
      frontier_center_marker.action = visualization_msgs::Marker::ADD;
      frontier_center_marker.lifetime = ros::Duration(2.0);
      frontier_center_marker.scale.x = frontier_center_marker.scale.y =
          frontier_center_marker.scale.z = 0.25;
      frontier_center_marker.color.r = 0.2;
      frontier_center_marker.color.g = 1.0;
      frontier_center_marker.color.b = 0.2;
      frontier_center_marker.color.a = 0.8;
      frontier_center_marker.pose.orientation.w = 1.0;

      // Per-node iteration: draw one line per (node → anchored-frontier)
      // pair. Color follows the node's type so anchored groups are visually
      // grouped — every frontier sharing a colour shares an owning node.
      for (const auto& node : nodes_) {
        if (!node.is_valid) continue;
        if (node.anchored_frontier_ids.empty()) continue;

        std_msgs::ColorRGBA color;
        if (node.type == TopologyNode::FRONTIER_ADJACENT) {
          color.r = 0.0; color.g = 0.8; color.b = 1.0;  // cyan
        } else {
          color.r = 1.0; color.g = 0.6; color.b = 0.0;  // orange
        }
        color.a = 0.85;

        geometry_msgs::Point node_p;
        node_p.x = node.position.x();
        node_p.y = node.position.y();
        node_p.z = 0.35;

        for (int fid : node.anchored_frontier_ids) {
          if (fid < 0 || fid >= static_cast<int>(frontier_averages.size()))
            continue;
          const auto& fpos = frontier_averages[fid];

          geometry_msgs::Point fp;
          fp.x = fpos.x();
          fp.y = fpos.y();
          fp.z = 0.3;
          frontier_center_marker.points.push_back(fp);

          link_marker.points.push_back(node_p);
          link_marker.points.push_back(fp);
          link_marker.colors.push_back(color);
          link_marker.colors.push_back(color);
        }
      }

      // Orphan frontiers (no node within anchor radius) — render as a
      // separate red SPHERE_LIST so they are visually distinct from
      // anchored ones. C-2 will fall back to legacy per-frontier scoring
      // for these.
      visualization_msgs::Marker orphan_marker;
      orphan_marker.header.frame_id = frame_id_;
      orphan_marker.header.stamp = ros::Time::now();
      orphan_marker.ns = "frontier_orphans";
      orphan_marker.id = 0;
      orphan_marker.type = visualization_msgs::Marker::SPHERE_LIST;
      orphan_marker.action = visualization_msgs::Marker::ADD;
      orphan_marker.lifetime = ros::Duration(2.0);
      orphan_marker.scale.x = orphan_marker.scale.y =
          orphan_marker.scale.z = 0.22;
      orphan_marker.color.r = 0.9;
      orphan_marker.color.g = 0.15;
      orphan_marker.color.b = 0.15;
      orphan_marker.color.a = 0.85;
      orphan_marker.pose.orientation.w = 1.0;
      for (const auto& op : orphan_frontiers_) {
        geometry_msgs::Point pt;
        pt.x = op.x();
        pt.y = op.y();
        pt.z = 0.30;
        orphan_marker.points.push_back(pt);
      }

      // Only push markers that have actual content — RViz flags empty
      // LINE_LIST / SPHERE_LIST as Topic Error. Lifetime=2s on previously
      // sent markers will auto-expire when we stop pushing.
      if (!link_marker.points.empty())
        link_markers.markers.push_back(link_marker);
      if (!frontier_center_marker.points.empty())
        link_markers.markers.push_back(frontier_center_marker);
      if (!orphan_marker.points.empty())
        link_markers.markers.push_back(orphan_marker);
    }

    ROS_INFO_THROTTLE(5.0, "[VoronoiTopology] Publishing %zu link markers", link_markers.markers.size());
    if (!link_markers.markers.empty()) node_frontier_link_pub_.publish(link_markers);
  }
}

// Agent control interface
void VoronoiTopology::setNodeAdditive(int node_id, double additive) {
  for (auto& node : nodes_) {
    if (node.id == node_id) {
      node.agent_additive = additive;
      return;
    }
  }
}

void VoronoiTopology::setNodeMultiplier(int node_id, double multiplier) {
  for (auto& node : nodes_) {
    if (node.id == node_id) {
      node.agent_multiplier = std::max(0.0, multiplier);
      return;
    }
  }
}

TopologyNode* VoronoiTopology::getNearestNode(const Eigen::Vector2d& pos) {
  TopologyNode* nearest = nullptr;
  double min_dist = std::numeric_limits<double>::max();

  for (auto& node : nodes_) {
    double dist = (node.position - pos).norm();
    if (dist < min_dist) {
      min_dist = dist;
      nearest = &node;
    }
  }
  return nearest;
}

// ==================== Memory Agent Interface ====================

std::vector<std::pair<int, double>> VoronoiTopology::getNearbyNodesWithMemory(
    const Eigen::Vector2d& pos, double radius, int max_nodes) const {

  std::vector<std::pair<int, double>> result;

  // Collect all nodes with memory within radius
  std::vector<std::pair<int, double>> candidates;
  for (const auto& node : nodes_) {
    double dist = (node.position - pos).norm();
    if (dist <= radius && node.hasMemory()) {
      candidates.push_back({node.id, dist});
    }
  }

  // Sort by distance
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Take top N
  for (size_t i = 0; i < std::min(static_cast<size_t>(max_nodes), candidates.size()); i++) {
    result.push_back(candidates[i]);
  }

  return result;
}

const TopologyNode* VoronoiTopology::getNodeById(int node_id) const {
  for (const auto& node : nodes_) {
    if (node.id == node_id) {
      return &node;
    }
  }
  return nullptr;
}

TopologyNode* VoronoiTopology::getNodeById(int node_id) {
  for (auto& node : nodes_) {
    if (node.id == node_id) {
      return &node;
    }
  }
  return nullptr;
}

}  // namespace skillnav_planner
