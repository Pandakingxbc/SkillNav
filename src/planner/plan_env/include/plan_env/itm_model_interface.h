#ifndef _ITM_MODEL_INTERFACE_H_
#define _ITM_MODEL_INTERFACE_H_

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::shared_ptr;

namespace skillnav_planner {

/**
 * @brief Abstract interface for Image-Text Matching (ITM) models
 *
 * This interface defines the contract for all ITM model implementations,
 * whether mock, CLIP-based, or custom VLM models.
 *
 * Design principles:
 * - Single Responsibility: Only compute image-text matching scores
 * - Interface Segregation: Minimal required methods
 * - Dependency Inversion: Depend on abstraction, not concrete implementations
 *
 * Usage:
 * ```cpp
 * shared_ptr<ITMModelInterface> itm_model = make_shared<CLIPITMModel>();
 * vector<double> scores = itm_model->computeITMScores(image, prompts);
 * ```
 *
 * Week 3: Support for real VLM models (CLIP, OpenCLIP)
 * Reference: docs/implementation_roadmap.md Week 3 Day 1-2
 */
class ITMModelInterface {
public:
    virtual ~ITMModelInterface() = default;

    /**
     * @brief Compute ITM scores for multiple text prompts
     *
     * This is the primary method that all implementations must provide.
     *
     * @param image Input RGB image (CV_8UC3)
     * @param prompts List of text prompts to match against the image
     * @return Vector of ITM scores [0.0, 1.0] for each prompt
     *
     * Contract:
     * - Output size must equal prompts.size()
     * - Scores must be in range [0.0, 1.0]
     * - Higher score means better match
     * - Thread-safe (can be called from multiple threads)
     */
    virtual vector<double> computeITMScores(
        const cv::Mat& image,
        const vector<string>& prompts) = 0;

    /**
     * @brief Compute single ITM score (convenience method)
     *
     * @param image Input RGB image
     * @param prompt Single text prompt
     * @return ITM score [0.0, 1.0]
     *
     * Default implementation calls computeITMScores() with single prompt.
     * Subclasses can override for optimization.
     */
    virtual double computeITMScore(
        const cv::Mat& image,
        const string& prompt) {
        vector<string> prompts = {prompt};
        vector<double> scores = computeITMScores(image, prompts);
        return scores.empty() ? 0.0 : scores[0];
    }

    /**
     * @brief Get model name/identifier
     *
     * @return String identifying the model (e.g., "CLIP-ViT-B/32", "ITMMock")
     */
    virtual string getModelName() const = 0;

    /**
     * @brief Check if model is ready for inference
     *
     * @return true if model loaded and ready, false otherwise
     */
    virtual bool isReady() const = 0;

    /**
     * @brief Get average inference latency in milliseconds
     *
     * @return Average latency for computeITMScores() call
     *
     * Used for monitoring and optimization.
     */
    virtual double getAverageLatency() const {
        return 0.0;  // Default: no tracking
    }

    /**
     * @brief Warmup model with dummy inference
     *
     * Some models (especially GPU-based) benefit from warmup to:
     * - Load weights into GPU memory
     * - Initialize CUDA kernels
     * - Avoid cold start penalty
     *
     * Default implementation: no-op
     */
    virtual void warmup() {
        // Default: do nothing
    }
};

/**
 * @brief Factory function type for creating ITM models
 *
 * Allows dependency injection and easy testing.
 */
using ITMModelFactory = std::function<shared_ptr<ITMModelInterface>()>;

}  // namespace skillnav_planner

#endif  // _ITM_MODEL_INTERFACE_H_
