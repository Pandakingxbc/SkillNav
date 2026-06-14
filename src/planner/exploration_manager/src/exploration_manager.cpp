/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <plan_env/map_ros.h>
#include <plan_env/voronoi_topology.h>
#include <path_searching/kino_astar.h>
#include <trajectory_manager/optimizer.h>
#include <ros/package.h>
#include <algorithm>

using namespace Eigen;

namespace skillnav_planner {

// Match ApexNav: default destruction. Members destruct in reverse declaration
// order (exploration_agent_ → memory_agent_ → multi_valuemap_manager_ → ...).
// Crash safety comes from lifetime-guard tokens inside Voronoi / MapROS,
// not from explicit ordering here.
ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(ros::NodeHandle& nh)
{
  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);
  sdf_map_->initMap(nh);
  object_map2d_ = sdf_map_->object_map2d_;

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, nh));
  path_finder_.reset(new Astar2D);
  path_finder_->init(nh, sdf_map_);

  // Connect frontier map to voronoi topology for visualization
  if (sdf_map_->voronoi_topology_) {
    sdf_map_->voronoi_topology_->setFrontierMap(frontier_map2d_);
    ROS_INFO("[ExplorationManager] Voronoi-Frontier link visualization enabled");
  }

  // Initialize Multi-ValueMap Manager
  multi_valuemap_manager_.reset(new MultiValueMapManager(sdf_map_.get(), nh));

  // Resolve prompt YAML path. Priority:
  //   1) exploration/prompt_yaml_path (explicit override)
  //   2) /target_object rosparam set by habitat_evaluation.py → <target>_prompts.yaml
  //   3) default: chair_prompts.yaml
  // habitat_evaluation sets /target_object before waiting for ROS to be ready,
  // but the planner may start first, so poll briefly.
  // Resolve <repo>/config/prompts/ relative to this package's install share dir.
  const string kPromptDir = ros::package::getPath("exploration_manager") + "/../../../config/prompts/";
  string prompt_yaml_path;
  if (!nh.getParam("exploration/prompt_yaml_path", prompt_yaml_path)) {
    string target;
    const double wait_sec = 60.0;
    ros::Time deadline = ros::Time::now() + ros::Duration(wait_sec);
    ROS_INFO("[ExplorationManager] Waiting up to %.0fs for /target_object…", wait_sec);
    while (ros::Time::now() < deadline) {
      if (nh.getParam("/target_object", target) && !target.empty()) break;
      ros::Duration(0.25).sleep();
    }
    if (!target.empty()) {
      // Sanitize for filesystem: convert spaces to underscores so targets like
      // "potted plant" map to "potted_plant_prompts.yaml".
      string fname_target = target;
      std::replace(fname_target.begin(), fname_target.end(), ' ', '_');
      prompt_yaml_path = kPromptDir + fname_target + "_prompts.yaml";
      ROS_INFO("[ExplorationManager] /target_object=%s → using %s",
               target.c_str(), prompt_yaml_path.c_str());
    } else {
      prompt_yaml_path = kPromptDir + "chair_prompts.yaml";
      ROS_WARN("[ExplorationManager] /target_object not set within %.0fs; falling back to %s",
               wait_sec, prompt_yaml_path.c_str());
    }
  }

  if (!multi_valuemap_manager_->loadPromptsFromYAML(prompt_yaml_path)) {
    ROS_ERROR("[ExplorationManager] Failed to load prompts from %s — falling back to chair_prompts.yaml so MVMM publishing stays alive",
              prompt_yaml_path.c_str());
    // CRITICAL: without a successful prompts load, value_maps_ stays empty and
    // /exploration/value_map_combined never publishes. Use chair_prompts.yaml
    // as a guaranteed-present fallback so the topic keeps flowing.
    const string fallback = kPromptDir + "chair_prompts.yaml";
    if (!multi_valuemap_manager_->loadPromptsFromYAML(fallback)) {
      ROS_FATAL("[ExplorationManager] Fallback chair_prompts.yaml also failed!");
    }
  } else {
    ROS_INFO("[ExplorationManager] Multi-ValueMap system initialized with prompts from %s",
             prompt_yaml_path.c_str());
  }

  // Hand the MVMM to MapROS so the depth/ITM data path can update both ValueMaps.
  if (sdf_map_ && sdf_map_->getMapROS()) {
    sdf_map_->getMapROS()->setMultiValueMapManager(multi_valuemap_manager_.get());
    ROS_INFO("[ExplorationManager] MapROS now feeds MultiValueMapManager");
  }

  // Voronoi nodes sample base_value from the fused map when MVMM is available.
  if (sdf_map_ && sdf_map_->voronoi_topology_) {
    sdf_map_->voronoi_topology_->setMultiValueMapManager(multi_valuemap_manager_.get());
  }

  // Initialize Memory Agent for candidate target management and FP tracking
  memory_agent_.reset(new MemoryAgent());
  memory_agent_->init(nh, sdf_map_, object_map2d_.get(),
                      sdf_map_->voronoi_topology_.get(), frontier_map2d_.get());
  ROS_INFO("[ExplorationManager] Memory Agent initialized");

  // Initialize Exploration Agent (rule-based SR/IG fusion-weight controller)
  exploration_agent_.reset(new ExplorationAgent());
  exploration_agent_->init(nh,
                           multi_valuemap_manager_.get(),
                           sdf_map_->voronoi_topology_.get(),
                           memory_agent_.get());
  ROS_INFO("[ExplorationManager] Exploration Agent initialized");

  // Initialize exploration data and parameter containers
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // Load exploration parameters from ROS parameter server
  nh.param("exploration/policy", ep_->policy_mode_, 0);
  nh.param("exploration/sigma_threshold", ep_->sigma_threshold_, 0.030);
  nh.param("exploration/max_to_mean_threshold", ep_->max_to_mean_threshold_, 1.2);
  nh.param("exploration/max_to_mean_percentage", ep_->max_to_mean_percentage_, 0.95);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  // HFTN policy knobs (used only when policy_mode_ == HFTN). Defaults
  // chosen so that node IG dominates for typical scenes (≤50 frontier
  // cells per node) and degree bonus adds modest preference for hub
  // nodes (a degree-4 junction gets ≈ 0.80 boost vs degree-1's ≈ 0.35).
  nh.param("exploration/hftn_degree_bonus", ep_->hftn_degree_bonus_, 0.5);
  nh.param("exploration/hftn_base_weight",  ep_->hftn_base_weight_,  1.0);
  nh.param("exploration/hftn_ig_scale",     ep_->hftn_ig_scale_,     50.0);

  // Get map parameters for ray casting initialization
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);
  tsp_client_ = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", true);

  // Initialize KinoAstar and GCopter for real-world trajectory planning
  kinoastar_.reset(new KinoAstar(nh, sdf_map_));
  kinoastar_->init();
  
  Config gcopter_config(nh);
  gcopter_.reset(new Gcopter(gcopter_config, nh, sdf_map_, kinoastar_));
  
  ROS_INFO("[ExplorationManager] KinoAstar and GCopter initialized for real-world mode");
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw)
{
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;

  // Update Memory Agent - sync candidates with ObjectMap2D
  if (memory_agent_) {
    memory_agent_->updateCandidates();
    // Check if any candidates need VLM verification
    memory_agent_->checkVerification(pos2d, yaw);
  }

  // Tick Exploration Agent (intermittent fusion-weight controller).
  // TODO: replace 0.5 with a real coverage estimate once exposed by FrontierMap2D.
  if (exploration_agent_) {
    exploration_agent_->tick(pos2d, 0.5);
  }

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================
  if (!object_clouds.empty()) {
    ROS_WARN_THROTTLE(5.0, "[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
        return SEARCH_BEST_OBJECT;
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN_THROTTLE(5.0, "[Navigation Mode (Over Depth)] Get over depth object cloud");
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (object_clouds.size() >= 1)
    top_object_cloud = object_clouds[0];

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    ROS_WARN("Maybe no passable frontier.");

    // Try suspicious objects as backup
    if (!top_object_cloud->points.empty() &&
        searchObjectPath(pos, top_object_cloud, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_SUSPICIOUS_OBJECT;
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("search exterme case!!!");

      // Try extreme object search with relaxed constraints
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Include lower confidence objects in extreme search
      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Try cached over-depth objects as final option
      static auto last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;
      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("No coverable frontier!!");
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("No passable frontier!!");
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Plan NBV] Total time %.2lf s too long!!!", total_time);

  return EXPLORATION;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] Find Closest Frontier");
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::SEMANTIC:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] Find Highest Semantic Value Frontier");
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HYBRID:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] Working on Hybrid Mode");
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::TSP_DIST:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] Working on TSP Distance Mode");
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HFTN:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] HFTN (Hybrid Frontier-Topology)");
      findHFTNPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    default:
      ROS_WARN_THROTTLE(5.0, "[Exploration Mode] Unknown Mode");
      break;
  }
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  double std_dev_threshold = ep_->sigma_threshold_;
  double max_to_mean_threshold = ep_->max_to_mean_threshold_;
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  double std_dev, max_to_mean, mean;
  calcSemanticFrontierInfo(sem_frontiers, std_dev, max_to_mean, mean);

  // Decide between exploitation and exploration based on semantic statistics
  if (std_dev > std_dev_threshold && max_to_mean > max_to_mean_threshold) {
    ROS_WARN_THROTTLE(5.0, "Exploit the semantic value (TSP)!!");
    vector<Vector2d> high_sem_frontiers;

    // Select high-value frontiers for TSP optimization
    for (auto sem_frontier : sem_frontiers) {
      double auto_max_to_mean_threshold =
          max(max_to_mean_threshold, ep_->max_to_mean_percentage_ * max_to_mean);
      if (sem_frontier.semantic_value / mean < auto_max_to_mean_threshold)
        break;
      high_sem_frontiers.push_back(sem_frontier.position);
    }
    findTSPTourPolicy(cur_pos, high_sem_frontiers, next_best_pos, next_best_path);
  }
  else {
    ROS_WARN_THROTTLE(5.0, "Explore the environment (Closest)!!");
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
  }
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Prefer the fused IG+SR combined value when MultiValueMapManager is available.
  // Fall back to the legacy single ValueMap otherwise.
  const bool use_mvm = (multi_valuemap_manager_ != nullptr);
  shared_ptr<ValueMap> legacy_vm = sdf_map_->value_map_;
  if (!use_mvm && !legacy_vm) {
    ROS_WARN_THROTTLE(5.0, "[ExplorationManager] No value map available for frontier scoring");
  }

  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    double value;
    if (use_mvm) {
      value = multi_valuemap_manager_->getCombinedValueAtGrid(idx);
      for (auto nbr : nbrs) value = max(value, multi_valuemap_manager_->getCombinedValueAtGrid(nbr));
    } else {
      value = legacy_vm ? legacy_vm->getValue(idx) : 0.0;
      if (legacy_vm) {
        for (auto nbr : nbrs) value = max(value, legacy_vm->getValue(nbr));
      }
    }

    frontier_values.emplace_back(frontier, value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Same failure-suppression cutoff as semantic policy. Kept here so closest-
  // frontier mode doesn't keep handing the planner a frontier in a direction
  // where exploration has already failed repeatedly.
  constexpr double kPassabilityCutoff = 0.2;
  auto* topology = sdf_map_->voronoi_topology_.get();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    if (topology) {
      auto* node = topology->getNearestNode(frontiers[i]);
      if (node && node->passability_multiplier < kPassabilityCutoff) continue;
    }

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
}

void ExplorationManager::findHFTNPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  // Hybrid Frontier-Topology Navigation (design memo C1, phase 2).
  //
  // Two-level decision built on the anchor data populated each Voronoi tick
  // by VoronoiTopology::computeFrontierAnchors():
  //   1. Score every node that owns at least one anchored frontier.
  //   2. From the highest-scoring node down, try to reach its anchored
  //      frontiers in order of per-cell semantic value.
  //   3. If no anchored frontier on any node is reachable, fall back to
  //      legacy semantic scoring on the orphan list, then on the full
  //      frontier set as ultimate safety net.

  next_best_path.clear();
  next_best_pos = cur_pos;  // safe default; replaced on success

  auto* topo = sdf_map_->voronoi_topology_.get();
  if (!topo || !frontier_map2d_) {
    findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
    return;
  }

  // Re-fetch frontier centroids so anchored_frontier_ids (which are indices
  // into FrontierMap2D's array) line up with our coordinate lookup.
  std::vector<std::vector<Vector2d>> frontier_clusters;
  std::vector<Vector2d> centroids;
  frontier_map2d_->getFrontiers(frontier_clusters, centroids);

  // ── Step 1: score and rank candidate nodes ─────────────────────────
  const double beta   = ep_->hftn_degree_bonus_;
  const double gamma  = ep_->hftn_base_weight_;
  const double scale  = std::max(1e-6, ep_->hftn_ig_scale_);

  struct ScoredNode { int node_id; double score; };
  std::vector<ScoredNode> scored;

  const auto& nodes = topo->getNodes();
  for (const auto& n : nodes) {
    if (!n.is_valid) continue;
    if (n.anchored_frontier_ids.empty()) continue;
    // Skip nodes that the Safe Agent has driven into the dead-zone floor.
    if (n.passability_multiplier < 0.2) continue;

    const double v_ig     = n.anchored_ig_sum / scale;       // normalised IG
    const double v_degree = std::log(1.0 + n.degree());       // hub bonus
    const double v_base   = n.base_value;                     // MVMM fused at cell

    double v_node = v_ig + beta * v_degree + gamma * v_base;
    v_node *= n.passability_multiplier * n.semantic_multiplier;

    scored.push_back({n.id, v_node});
  }

  std::sort(scored.begin(), scored.end(),
            [](const ScoredNode& a, const ScoredNode& b) { return a.score > b.score; });

  // Helper: id -> node lookup. nodes_ is a vector keyed by index not id;
  // VoronoiTopology already has getNearestNode but not getNodeById... it
  // actually does via VoronoiTopology::getNodeById per the header. Use it.

  // ── Step 2: walk sorted nodes, try their anchored frontiers ────────
  // Within a node, sort its anchored frontiers by per-cell semantic value
  // (mirroring findHighestSemanticsFrontierPolicy).
  const bool use_mvm = (multi_valuemap_manager_ != nullptr);
  std::shared_ptr<ValueMap> legacy_vm = sdf_map_->value_map_;

  auto scoreFrontier = [&](const Vector2d& fpos) -> double {
    Vector2i idx;
    sdf_map_->posToIndex(fpos, idx);
    if (use_mvm) return multi_valuemap_manager_->getCombinedValueAtGrid(idx);
    if (legacy_vm)  return legacy_vm->getValue(idx);
    return 0.0;
  };

  for (const auto& sn : scored) {
    auto* node = topo->getNodeById(sn.node_id);
    if (node == nullptr) continue;

    // Build sorted local-frontier list for this node.
    std::vector<std::pair<int, double>> local;
    local.reserve(node->anchored_frontier_ids.size());
    for (int fid : node->anchored_frontier_ids) {
      if (fid < 0 || fid >= static_cast<int>(centroids.size())) continue;
      local.emplace_back(fid, scoreFrontier(centroids[fid]));
    }
    std::sort(local.begin(), local.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& pr : local) {
      const Vector2d& fpos = centroids[pr.first];
      std::vector<Vector2d> tmp_path;
      Vector2d tmp_pos;
      if (!searchFrontierPath(cur_pos, fpos, tmp_pos, tmp_path)) continue;
      next_best_pos  = tmp_pos;
      next_best_path = tmp_path;
      ROS_INFO_THROTTLE(3.0,
          "[HFTN] picked node %d (score=%.2f, %zu anchored), frontier f=%d at (%.2f,%.2f)",
          sn.node_id, sn.score, node->anchored_frontier_ids.size(),
          pr.first, fpos.x(), fpos.y());
      return;
    }
  }

  // ── Step 3: orphan fallback ────────────────────────────────────────
  const auto& orphans = topo->getOrphanFrontiers();
  if (!orphans.empty()) {
    std::vector<Vector2d> orphan_vec(orphans.begin(), orphans.end());
    findHighestSemanticsFrontierPolicy(cur_pos, orphan_vec, next_best_pos, next_best_path);
    if (!next_best_path.empty()) {
      ROS_INFO_THROTTLE(3.0, "[HFTN] orphan-fallback chose (%.2f,%.2f), %zu orphans",
                        next_best_pos.x(), next_best_pos.y(), orphans.size());
      return;
    }
  }

  // ── Step 4: ultimate fallback — legacy semantic on the full set ────
  findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
  ROS_WARN_THROTTLE(5.0, "[HFTN] full-frontier fallback used (%zu candidates)",
                    frontiers.size());
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    ROS_ERROR("No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize ATSP par file
  // Create problem file
  ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  auto par_dir = ep_->tsp_dir_ + "/atsp_tour.atsp";

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/atsp_tour.tour");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN_THROTTLE(5.0, "[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    ROS_ERROR("[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = object_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        ROS_WARN("%s", debug_msg.c_str());
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  const double max_search_time = 0.2;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Find nearest accessible point in object cloud
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error indicator from findNearestObjectPoint

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85 };
  const std::vector<std::string> debug_messages = { "I'm going to the object! dist = 0.5m!",
    "I'm going to the object! dist = 0.70m!", "I'm going to the object! dist = 0.85m!" };

  // Attempt path planning with each safety distance
  for (size_t i = 0; i < distances.size(); ++i) {
    if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
            refined_pos, refined_path, debug_messages[i])) {
      return true;
    }
  }

  ROS_ERROR("Failed to find object path.");
  return false;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on semantic values and reachability
  sem_frontiers.clear();

  // Prefer fused IG+SR combined value; fall back to legacy single ValueMap.
  const bool use_mvm = (multi_valuemap_manager_ != nullptr);
  shared_ptr<ValueMap> legacy_vm = sdf_map_->value_map_;

  auto sample_value = [&](const Vector2i& at) -> double {
    if (use_mvm) return multi_valuemap_manager_->getCombinedValueAtGrid(at);
    return legacy_vm ? legacy_vm->getValue(at) : 0.0;
  };

  // Frontier failure suppression: skip frontiers whose nearest Voronoi node has
  // been repeatedly marked dormant / VLM-confirmed dead. passability_multiplier
  // is driven by SafeAgent (VLM) and by FSM dormant-frontier hooks; threshold
  // 0.2 corresponds to ~3 consecutive failures with our 0.3x penalty step.
  constexpr double kPassabilityCutoff = 0.2;
  auto* topology = sdf_map_->voronoi_topology_.get();

  for (auto& frontier : frontiers) {
    if (topology) {
      auto* node = topology->getNearestNode(frontier);
      if (node && node->passability_multiplier < kPassabilityCutoff) continue;
    }

    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double value = sample_value(idx);

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(idx) == 1 ||
          sdf_map_->getOccupancy(idx) == SDFMap2D::OCCUPIED)
        continue;
      value = std::max(value, sample_value(nbr));
    }
    sem_frontier.semantic_value = value;

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

bool ExplorationManager::planTrajectory(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl)
{
  if (!gcopter_ || !kinoastar_) {
    ROS_WARN_THROTTLE(1.0, "[ExplorationManager] GCopter or KinoAstar not initialized for real-world mode");
    return false;
  }
  
  Eigen::VectorXd goal_state, current_state;
  Vector3d control = ctrl;
  goal_state = end;
  current_state = start;

  // Kinodynamic A* search
  kinoastar_->reset();
  kinoastar_->search(goal_state, current_state, control);
  kinoastar_->getKinoNode();
  
  if (kinoastar_->has_path_) {
    kinoastar_->kinoastarFlatPathPub(kinoastar_->flat_trajs_);
    gcopter_->minco_plan();
    std::vector<Trajectory<7, 3>> final_trajes = gcopter_->final_trajes;
    gcopter_->mincoPathPub(gcopter_->final_trajes, gcopter_->final_singuls);
    return true;
  }
  
  return false;
}

}  // namespace skillnav_planner
