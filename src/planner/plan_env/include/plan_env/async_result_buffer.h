#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <ros/ros.h>

namespace skillnav_planner {

/**
 * AsyncResultBuffer<T> — single-slot mailbox between ROS callback and agent loop.
 *
 * Motivation
 * ----------
 * SkillNav's agents talk to LLM / VLM services via ROS pub-sub. The request is
 * fired by the agent and the response arrives asynchronously on the ROS
 * callback thread. The agent loop ticks at a fixed cadence and must *never*
 * block on the response — it has to keep producing actions for the FSM. The
 * canonical pattern is therefore:
 *
 *   - When the agent decides to ask: publish request, mark inflight.
 *   - When the response callback fires: deposit result into a buffer, mark new.
 *   - On the next agent tick: non-blocking try_consume; if a new result is
 *     available, apply its effect (update Voronoi modifiers, etc.) and reset.
 *
 * This template captures that pattern once so SafeAgent / MemoryAgent /
 * StrategicAgent can share the same shape with different T.
 *
 * Concurrency
 * -----------
 * ROS Noetic's default spinner runs callbacks on the same thread as ros::spin,
 * so by default markFired() and provide() are not racing — but
 * MultiThreadedSpinner is supported and someone might switch. The mutex makes
 * the buffer correct under either model with negligible overhead.
 *
 * Single-slot semantics
 * ---------------------
 * If a new request is fired while a previous one is still inflight, the older
 * result (if it ever arrives) is silently dropped. This matches the intent for
 * navigation: a stale VLM judgment about a position the robot has long since
 * left is worse than no judgment.
 */
template <typename T>
class AsyncResultBuffer {
 public:
  AsyncResultBuffer() = default;

  /// Call right after publishing the ROS request. Clears any stale result and
  /// records fire time so the agent can age-out lost responses if it wants.
  void markFired() {
    std::lock_guard<std::mutex> lock(mutex_);
    fired_at_ = ros::Time::now();
    inflight_.store(true);
    has_new_.store(false);
  }

  /// Call from the ROS subscriber callback to deposit a parsed result.
  /// Drops the result if no fire is currently inflight (stale response).
  void provide(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inflight_.load()) return;  // Stale response — fire was cancelled.
    value_ = value;
    inflight_.store(false);
    has_new_.store(true);
  }

  /// Non-blocking consume. If a new result is available, writes it to `out`
  /// and returns true (also clears the buffer so subsequent calls return
  /// false until another provide). Returns false if no new result.
  bool tryConsume(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_new_.load()) return false;
    out = value_;
    has_new_.store(false);
    return true;
  }

  /// Discard the inflight context — e.g. the agent moved to a different region
  /// before the response arrived, so the response would be misleading even if
  /// it eventually shows up. Future provide() calls become no-ops until next
  /// markFired().
  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_.store(false);
    has_new_.store(false);
  }

  /// Whether a request is currently outstanding (fired but not provided).
  bool isInflight() const { return inflight_.load(); }

  /// Whether tryConsume would return a result.
  bool hasNew() const { return has_new_.load(); }

  /// Time since the most recent markFired(), or zero if not inflight.
  /// Useful for the agent to age-out abandoned requests.
  ros::Duration inflightAge() const {
    if (!inflight_.load()) return ros::Duration(0);
    return ros::Time::now() - fired_at_;
  }

 private:
  mutable std::mutex mutex_;
  std::atomic<bool> inflight_{false};
  std::atomic<bool> has_new_{false};
  T value_;
  ros::Time fired_at_;
};

/**
 * AsyncResultQueue<T> — multi-slot variant of AsyncResultBuffer.
 *
 * Use when more than one request can be inflight simultaneously and you can't
 * afford to drop late responses. The intended caller is MemoryAgent's
 * candidate verification: N candidates can each have one VLM verify call
 * outstanding, and the responses for different candidates carry distinct
 * routing keys (cluster_id / anchor_node_id) inside the payload, so dropping
 * is wrong.
 *
 * Mental model
 *   - The agent's request side calls markFired() once per request published.
 *     The internal `inflight_count_` increments.
 *   - The ROS callback calls provide(value) per response. If a marker is
 *     outstanding, the value is appended to the queue and the counter
 *     decrements. Responses that arrive with no outstanding markers (stale /
 *     duplicate) are silently dropped.
 *   - The agent loop calls tryConsume(out) once per tick *in a loop* until it
 *     returns false, draining every pending result. Each consumed result is
 *     routed (by payload) to whichever entity it belongs to.
 *
 * Routing is the caller's job — this template only ensures fan-in correctness.
 */
template <typename T>
class AsyncResultQueue {
 public:
  AsyncResultQueue() = default;

  /// Call right after publishing one request. Bumps the in-flight counter.
  void markFired() {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_count_++;
  }

  /// Call from the ROS subscriber callback. Append result to the queue,
  /// matched against an outstanding marker. Stale / duplicate responses
  /// (no outstanding marker) are silently dropped.
  void provide(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inflight_count_ == 0) return;
    inflight_count_--;
    queue_.push_back(value);
  }

  /// Non-blocking pop of the oldest pending result.
  /// Returns false (and does not modify out) when the queue is empty.
  /// Recommended idiom on the agent loop:
  ///   T r; while (queue_.tryConsume(r)) { applyTo(r); }
  bool tryConsume(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    out = queue_.front();
    queue_.pop_front();
    return true;
  }

  /// Drop everything — pending results and outstanding markers.
  void cancelAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_count_ = 0;
    queue_.clear();
  }

  /// How many requests have been markFired() but not yet matched with a provide.
  size_t inflightCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return inflight_count_;
  }

  /// How many provide()d results are waiting to be consumed.
  size_t pendingResults() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  size_t inflight_count_{0};
  std::deque<T> queue_;
};

}  // namespace skillnav_planner
