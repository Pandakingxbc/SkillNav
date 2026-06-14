#include <plan_env/multi_valuemap_manager.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iomanip>
#include <sstream>

namespace skillnav_planner {

MultiValueMapManager::MultiValueMapManager(SDFMap2D* sdf_map, ros::NodeHandle& nh)
    : fusion_weights_(NUM_MAPS, 0.5),
      object_map_prompt_id_(SR_ID),
      last_target_itm_score_(0.0),
      sdf_map_(sdf_map),
      nh_(nh) {
    ROS_INFO("[MultiValueMapManager] Initializing (two-map fusion mode)");
    initializePublishers();
}

MultiValueMapManager::~MultiValueMapManager() {
    ROS_INFO("[MultiValueMapManager] Shutting down");
    // Defensive fallback — preferred path is FSM calling clearVisualization()
    // before destruction so the publishers are still alive long enough for the
    // messages to drain.
    clearVisualization();
}

void MultiValueMapManager::clearVisualization() {
    if (!ros::ok()) return;
    if (combined_value_pub_) {
        // RViz ignores width=0 PointCloud2 (treats as "no update"), so send a
        // single point parked far below the scene. With Decay Time=0 in RViz
        // this REPLACES the previous cloud so the visualization is cleared.
        pcl::PointCloud<pcl::PointXYZI> kick;
        pcl::PointXYZI pt;
        pt.x = pt.y = pt.z = -1000.0f;
        pt.intensity = 0.0f;
        kick.points.push_back(pt);
        kick.width = 1;
        kick.height = 1;
        kick.is_dense = true;
        kick.header.frame_id = "world";
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(kick, msg);
        combined_value_pub_.publish(msg);
    }
    if (fusion_info_marker_pub_) {
        visualization_msgs::Marker del;
        del.header.frame_id = "world";
        del.header.stamp = ros::Time::now();
        del.ns = "fusion_info";
        del.id = 0;
        del.action = visualization_msgs::Marker::DELETEALL;
        fusion_info_marker_pub_.publish(del);
    }
}

// ===== Initialization =====

bool MultiValueMapManager::loadPromptsFromYAML(const string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);

        prompts_.clear();
        value_maps_.clear();

        string target_object = config["target_object"].as<string>();
        ROS_INFO("[MultiValueMapManager] Loading prompts for target: %s", target_object.c_str());

        YAML::Node prompts_node = config["prompts"];
        for (size_t i = 0; i < prompts_node.size(); ++i) {
            Prompt prompt;
            prompt.id = prompts_node[i]["id"].as<int>();
            prompt.type = prompts_node[i]["type"].as<string>();
            prompt.text = prompts_node[i]["text"].as<string>();
            if (prompts_node[i]["hypothesis"]) {
                prompt.hypothesis = prompts_node[i]["hypothesis"].as<string>();
            }
            if (prompts_node[i]["initial_weight"]) {
                prompt.initial_weight = prompts_node[i]["initial_weight"].as<double>();
            }
            prompts_.push_back(prompt);
            ROS_INFO("[MultiValueMapManager] Loaded Prompt %d: %s (initial_weight=%.2f)",
                     prompt.id, prompt.type.c_str(), prompt.initial_weight);
        }

        if (config["object_map_prompt_id"]) {
            object_map_prompt_id_ = config["object_map_prompt_id"].as<int>();
        }

        if (!validatePrompts()) {
            ROS_ERROR("[MultiValueMapManager] Prompt validation failed");
            return false;
        }

        for (size_t i = 0; i < prompts_.size(); ++i) {
            value_maps_.push_back(std::make_shared<ValueMap>(sdf_map_, nh_));
        }

        // Seed fusion weights from prompts' initial_weight
        fusion_weights_[SR_ID] = prompts_[SR_ID].initial_weight;
        fusion_weights_[IG_ID] = prompts_[IG_ID].initial_weight;
        normalizeFusionWeights();

        ROS_INFO("[MultiValueMapManager] Loaded %zu prompts. ObjectMap uses prompt id=%d",
                 prompts_.size(), object_map_prompt_id_);
        ROS_INFO("[MultiValueMapManager] Initial fusion weights: SR=%.2f, IG=%.2f",
                 fusion_weights_[SR_ID], fusion_weights_[IG_ID]);
        return true;

    } catch (const YAML::Exception& e) {
        ROS_ERROR("[MultiValueMapManager] YAML error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        ROS_ERROR("[MultiValueMapManager] Error: %s", e.what());
        return false;
    }
}

// ===== Update =====

void MultiValueMapManager::updateAllValueMaps(const Vector2d& sensor_pos,
                                              const double& sensor_yaw,
                                              const vector<Vector2i>& free_grids,
                                              const vector<double>& itm_scores) {
    if (static_cast<int>(itm_scores.size()) != NUM_MAPS) {
        ROS_ERROR("[MultiValueMapManager] Expected %d ITM scores, got %zu",
                  NUM_MAPS, itm_scores.size());
        return;
    }
    if (static_cast<int>(value_maps_.size()) != NUM_MAPS) {
        ROS_WARN_THROTTLE(5.0, "[MultiValueMapManager] ValueMaps not initialized");
        return;
    }

    for (int i = 0; i < NUM_MAPS; ++i) {
        value_maps_[i]->updateValueMap(sensor_pos, sensor_yaw, free_grids, itm_scores[i]);
        prompts_[i].updateStats(itm_scores[i]);
    }
    last_target_itm_score_ = itm_scores[object_map_prompt_id_];

    ROS_DEBUG("[MultiValueMapManager] Updated maps. SR=%.3f IG=%.3f weights=[%.2f, %.2f]",
              itm_scores[SR_ID], itm_scores[IG_ID],
              fusion_weights_[SR_ID], fusion_weights_[IG_ID]);
}

// ===== Query =====

shared_ptr<ValueMap> MultiValueMapManager::getValueMap(int id) const {
    if (id < 0 || id >= static_cast<int>(value_maps_.size())) {
        ROS_ERROR("[MultiValueMapManager] Invalid map id: %d", id);
        return nullptr;
    }
    return value_maps_[id];
}

// ===== Fusion =====

void MultiValueMapManager::setFusionWeights(double w_sr, double w_ig) {
    fusion_weights_[SR_ID] = std::max(0.0, w_sr);
    fusion_weights_[IG_ID] = std::max(0.0, w_ig);
    normalizeFusionWeights();
    ROS_INFO("[MultiValueMapManager] Fusion weights set: SR=%.3f, IG=%.3f",
             fusion_weights_[SR_ID], fusion_weights_[IG_ID]);
}

void MultiValueMapManager::normalizeFusionWeights() {
    double sum = fusion_weights_[SR_ID] + fusion_weights_[IG_ID];
    if (sum < 1e-6) {
        fusion_weights_[SR_ID] = 0.5;
        fusion_weights_[IG_ID] = 0.5;
        return;
    }
    fusion_weights_[SR_ID] /= sum;
    fusion_weights_[IG_ID] /= sum;
}

double MultiValueMapManager::getCombinedValue(const Vector2d& pos) const {
    if (value_maps_.size() != NUM_MAPS) return 0.0;
    double v_sr = value_maps_[SR_ID]->getValue(pos);
    double v_ig = value_maps_[IG_ID]->getValue(pos);
    return fusion_weights_[SR_ID] * v_sr + fusion_weights_[IG_ID] * v_ig;
}

double MultiValueMapManager::getCombinedValueAtGrid(const Vector2i& grid) const {
    if (value_maps_.size() != NUM_MAPS) return 0.0;
    double v_sr = value_maps_[SR_ID]->getValue(grid);
    double v_ig = value_maps_[IG_ID]->getValue(grid);
    return fusion_weights_[SR_ID] * v_sr + fusion_weights_[IG_ID] * v_ig;
}

// ===== Visualization =====

void MultiValueMapManager::publishVisualization() {
    if (value_maps_.size() != NUM_MAPS) return;

    sensor_msgs::PointCloud2 cloud;
    generateCombinedPointCloud(cloud);
    combined_value_pub_.publish(cloud);

    // Info marker showing prompts and weights
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.ns = "fusion_info";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.position.x = 0.0;
    text_marker.pose.position.y = 0.0;
    text_marker.pose.position.z = 2.5;
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = 0.3;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    std::stringstream ss;
    ss << "Fusion: SR=" << std::fixed << std::setprecision(2)
       << fusion_weights_[SR_ID] << " IG=" << fusion_weights_[IG_ID];
    text_marker.text = ss.str();
    fusion_info_marker_pub_.publish(text_marker);
}

void MultiValueMapManager::generateCombinedPointCloud(sensor_msgs::PointCloud2& cloud) const {
    std::vector<Vector2d> positions;
    std::vector<double> values;

    // Iterate the cumulative explored bounds to show the full mapped area.
    Eigen::Vector2i bmin_idx, bmax_idx;
    sdf_map_->getUpdateBoundIndices(bmin_idx, bmax_idx);
    sdf_map_->boundIndex(bmin_idx);
    sdf_map_->boundIndex(bmax_idx);

    // Filter on SR (semantic relevance) alone for visibility, since IG
    // (information gain) gets non-zero values on every observed cell and
    // would make the fused map cover everything. The DISPLAYED value still
    // uses the fused weight — only the "should this cell be rendered" gate
    // mirrors the legacy /grid_map/value_map behavior (which uses the SR-
    // equivalent single ITM score).
    for (int x = bmin_idx.x(); x <= bmax_idx.x(); ++x) {
        for (int y = bmin_idx.y(); y <= bmax_idx.y(); ++y) {
            Eigen::Vector2i idx(x, y);
            double v_sr = value_maps_[SR_ID]->getValue(idx);
            if (v_sr <= 1e-3) continue;  // gate by SR like the legacy display

            if (!sdf_map_->isFreeCell(idx)) continue;

            double v_ig = value_maps_[IG_ID]->getValue(idx);
            double val = fusion_weights_[SR_ID] * v_sr + fusion_weights_[IG_ID] * v_ig;

            Vector2d pos;
            sdf_map_->indexToPos(idx, pos);
            positions.push_back(pos);
            values.push_back(val);
        }
    }

    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl_cloud.points.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        pcl::PointXYZI pt;
        pt.x = positions[i].x();
        pt.y = positions[i].y();
        pt.z = 0.1;
        pt.intensity = static_cast<float>(std::min(1.0, std::max(0.0, values[i])));
        pcl_cloud.points.push_back(pt);
    }
    pcl_cloud.width = pcl_cloud.points.size();
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = true;
    pcl_cloud.header.frame_id = "world";
    pcl::toROSMsg(pcl_cloud, cloud);
}

// ===== Stats =====

void MultiValueMapManager::printStatistics() const {
    ROS_INFO("=== MultiValueMapManager (two-map fusion) ===");
    ROS_INFO("Fusion weights: SR=%.3f, IG=%.3f",
             fusion_weights_[SR_ID], fusion_weights_[IG_ID]);
    for (size_t i = 0; i < prompts_.size(); ++i) {
        ROS_INFO("Map %zu [%s]: used=%d avg=%.3f",
                 i, prompts_[i].type.c_str(),
                 prompts_[i].usage_count, prompts_[i].avg_score);
    }
}

void MultiValueMapManager::resetStatistics() {
    for (auto& p : prompts_) p.resetStats();
}

// ===== Helpers =====

bool MultiValueMapManager::validatePrompts() const {
    if (static_cast<int>(prompts_.size()) != NUM_MAPS) {
        ROS_ERROR("[MultiValueMapManager] Expected %d prompts, got %zu",
                  NUM_MAPS, prompts_.size());
        return false;
    }
    if (prompts_[SR_ID].id != SR_ID || prompts_[SR_ID].type != "semantic_relevance") {
        ROS_ERROR("[MultiValueMapManager] Prompt 0 must be id=0 type=semantic_relevance "
                  "(got id=%d type=%s)", prompts_[SR_ID].id, prompts_[SR_ID].type.c_str());
        return false;
    }
    if (prompts_[IG_ID].id != IG_ID || prompts_[IG_ID].type != "information_gain") {
        ROS_ERROR("[MultiValueMapManager] Prompt 1 must be id=1 type=information_gain "
                  "(got id=%d type=%s)", prompts_[IG_ID].id, prompts_[IG_ID].type.c_str());
        return false;
    }
    if (object_map_prompt_id_ != SR_ID) {
        ROS_WARN("[MultiValueMapManager] object_map_prompt_id=%d, expected %d (SR). "
                 "Overriding to SR.", object_map_prompt_id_, SR_ID);
        const_cast<int&>(object_map_prompt_id_) = SR_ID;
    }
    return true;
}

void MultiValueMapManager::initializePublishers() {
    combined_value_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/exploration/value_map_combined", 1);
    fusion_info_marker_pub_ = nh_.advertise<visualization_msgs::Marker>(
        "/exploration/fusion_info", 10);
    // Periodically publish the fused map so RViz always has fresh data.
    vis_timer_ = nh_.createTimer(ros::Duration(1.0),
        &MultiValueMapManager::visTimerCallback, this);
    ROS_INFO("[MultiValueMapManager] Publishers initialized (combined-only, 1Hz)");
}

void MultiValueMapManager::visTimerCallback(const ros::TimerEvent&) {
    if (value_maps_.size() != NUM_MAPS) return;
    publishVisualization();
}

}  // namespace skillnav_planner
