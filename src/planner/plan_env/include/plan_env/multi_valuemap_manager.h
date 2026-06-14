#ifndef _MULTI_VALUEMAP_MANAGER_H_
#define _MULTI_VALUEMAP_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <sensor_msgs/PointCloud2.h>

#include <plan_env/value_map2d.h>
#include <plan_env/sdf_map2d.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::string;

namespace skillnav_planner {

/**
 * @brief Prompt structure for two-map fusion system
 *
 * Exactly two prompts are supported:
 *   - SR (semantic_relevance): target-direct ITM, drives ObjectMap confidence
 *   - IG (information_gain): exploration-oriented ITM, drives broad coverage
 */
struct Prompt {
    int id;
    string type;          // "semantic_relevance" | "information_gain"
    string text;          // ITM prompt text
    string hypothesis;    // Optional rationale
    double initial_weight = 0.5;

    // Runtime statistics
    int usage_count = 0;
    double avg_score = 0.0;
    double cumulative_score = 0.0;

    void resetStats() {
        usage_count = 0;
        avg_score = 0.0;
        cumulative_score = 0.0;
    }

    void updateStats(double score) {
        usage_count++;
        cumulative_score += score;
        avg_score = cumulative_score / usage_count;
    }
};

/**
 * @brief Two-map ValueMap manager with weighted fusion (no switching).
 *
 * Architecture (post-refactor):
 *   - Maintains exactly 2 ValueMaps: SR (semantic) + IG (information gain)
 *   - Updates both on every observation using two ITM scores
 *   - Exposes `getCombinedValue` as weighted fusion (w_ig * IG + w_sr * SR)
 *   - Fusion weights are normalized to sum=1 and controlled by ExplorationAgent
 *   - Only SR map's ITM score drives ObjectMap confidence (target-direct)
 */
class MultiValueMapManager {
public:
    static constexpr int NUM_MAPS = 2;
    static constexpr int SR_ID = 0;  ///< semantic_relevance (target-direct)
    static constexpr int IG_ID = 1;  ///< information_gain

    MultiValueMapManager(SDFMap2D* sdf_map, ros::NodeHandle& nh);
    ~MultiValueMapManager();

    /// Publish empty cloud + DELETEALL marker so RViz drops the previous
    /// episode's fused map immediately. Safe to call from FSM init() while
    /// publishers are still alive — the dtor variant is fragile because
    /// publisher destruction can drop in-flight ROS messages.
    void clearVisualization();

    // ===== Initialization =====

    /**
     * @brief Load 2 prompts from YAML file.
     *
     * YAML format:
     *   target_object: "toilet"
     *   prompts:
     *     - id: 0
     *       type: "semantic_relevance"
     *       text: "..."
     *       initial_weight: 0.5
     *     - id: 1
     *       type: "information_gain"
     *       text: "..."
     *       initial_weight: 0.5
     *   object_map_prompt_id: 0
     */
    bool loadPromptsFromYAML(const string& yaml_path);

    int getNumMaps() const { return value_maps_.size(); }

    // ===== Update =====

    /**
     * @brief Update both ValueMaps with their corresponding ITM scores.
     * @param itm_scores size must equal NUM_MAPS; itm_scores[SR_ID] is target, [IG_ID] is exploration.
     */
    void updateAllValueMaps(const Vector2d& sensor_pos,
                            const double& sensor_yaw,
                            const vector<Vector2i>& free_grids,
                            const vector<double>& itm_scores);

    // ===== Query =====

    /** @brief Get one ValueMap by id (SR_ID or IG_ID). */
    shared_ptr<ValueMap> getValueMap(int id) const;

    /** @brief Get the SR (target) ITM score most recently seen. Used by ObjectMap. */
    double getLastTargetITMScore() const { return last_target_itm_score_; }

    /** @brief Get the prompt id that should drive ObjectMap (always SR by design). */
    int getObjectMapPromptId() const { return object_map_prompt_id_; }

    /** @brief All prompts (size = 2). */
    const vector<Prompt>& getPrompts() const { return prompts_; }

    // ===== Fusion =====

    /**
     * @brief Set fusion weights. Auto-normalized to sum=1.
     *        Negative values are clamped to 0. If both 0, falls back to [0.5, 0.5].
     */
    void setFusionWeights(double w_sr, double w_ig);

    /** @brief Current fusion weights (normalized). */
    void getFusionWeights(double& w_sr, double& w_ig) const {
        w_sr = fusion_weights_[SR_ID];
        w_ig = fusion_weights_[IG_ID];
    }

    /** @brief Combined value at a position: w_sr * SR(pos) + w_ig * IG(pos). */
    double getCombinedValue(const Vector2d& pos) const;

    /** @brief Combined value at a grid index. */
    double getCombinedValueAtGrid(const Vector2i& grid) const;

    // ===== Visualization =====

    /** @brief Publish only the combined (fused) value map as PointCloud2. */
    void publishVisualization();

    // ===== Stats =====

    void printStatistics() const;
    void resetStatistics();

private:
    // ===== Data Members =====

    vector<Prompt> prompts_;                         ///< exactly 2 prompts
    vector<shared_ptr<ValueMap>> value_maps_;        ///< exactly 2 ValueMaps
    vector<double> fusion_weights_;                  ///< [w_sr, w_ig], sum=1

    int object_map_prompt_id_;                       ///< id that drives ObjectMap (default SR_ID)
    double last_target_itm_score_;                   ///< last SR ITM score

    SDFMap2D* sdf_map_;
    ros::NodeHandle nh_;

    // ===== ROS Publishers / Timers =====

    ros::Publisher combined_value_pub_;              ///< /exploration/value_map_combined
    ros::Publisher fusion_info_marker_pub_;          ///< debug: prompt + weight text
    ros::Timer vis_timer_;                           ///< periodically publishes combined map
    void visTimerCallback(const ros::TimerEvent&);

    // ===== Helpers =====

    bool validatePrompts() const;
    void initializePublishers();
    void generateCombinedPointCloud(sensor_msgs::PointCloud2& cloud) const;
    void normalizeFusionWeights();
};

}  // namespace skillnav_planner

#endif  // _MULTI_VALUEMAP_MANAGER_H_
