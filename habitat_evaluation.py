"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import atexit
import gzip
import json
import os
import signal
import subprocess
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64, Float64MultiArray
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from llm.instrumentation import set_episode as _instr_set_episode
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine, get_itm_message_cosines_dual
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


_planner_proc = None
_PLANNER_PROBE_TOPIC = "/grid_map/occupied"
_PLANNER_LOG_PATH = "/tmp/skillnav_planner.log"


def _kill_planner_proc():
    """Terminate the planner process group started by restart_planner (if any)."""
    global _planner_proc
    if _planner_proc is None or _planner_proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(_planner_proc.pid), signal.SIGTERM)
        try:
            _planner_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(_planner_proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def restart_planner(timeout=60.0):
    """Kill any running exploration_node and relaunch exploration_multiagent.launch.

    Blocks until /grid_map/occupied is advertised by the new planner, or raises
    RuntimeError on timeout. This is the per-episode hard-reset that wipes
    SDFMap / FrontierMap / VoronoiTopology state from the previous scene.
    """
    global _planner_proc

    print("[restart_planner] killing previous planner ...")
    _kill_planner_proc()
    # Belt-and-suspenders: also kill anything matching the patterns (covers
    # planners launched manually in a separate terminal before this script ran).
    # SIGKILL (-9): skip dtors so the old 1Hz Voronoi / MVMM timers stop
    # firing IMMEDIATELY, otherwise they keep publishing the previous episode's
    # markers during the relaunch window and RViz oscillates between old/new.
    # Lifetime safety (weak_ptr + tracked_object) makes skipped dtors safe.
    #
    # Kill the roslaunch parent FIRST. A stale `roslaunch exploration_multiagent`
    # left over from a previous session (e.g. a manual launch in another terminal)
    # would otherwise re-spawn its children the moment we pkill them, and the
    # respawn races with our own new roslaunch — losing nodes to "new node
    # registered with same name" shutdowns and stalling /grid_map/occupied.
    for pattern in ("roslaunch.*exploration_multiagent",
                    "exploration_node",
                    "vlm_safe_agent_node",
                    "vlm_memory_agent_node",
                    "strategic_agent_node"):
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    # Wait until ALL old processes are actually gone before launching the new
    # one — INCLUDING the Python agent nodes, otherwise their stale rosmaster
    # registrations cause "new node registered with same name" → old gets
    # kicked → cascade shutdown → habitat stuck on odometry. Also do
    # `rosnode cleanup` to drop phantom registrations left by crashed sessions.
    for _ in range(50):  # up to 5s
        leftover = subprocess.run(
            ["pgrep", "-f",
             "exploration_node|roslaunch.*exploration_multiagent|"
             "vlm_memory_agent_node|vlm_safe_agent_node|strategic_agent_node"],
            capture_output=True)
        if leftover.returncode != 0:
            break
        time.sleep(0.1)
    # Drop phantom registrations left in rosmaster from any prior crashed
    # session. `rosnode cleanup` interactive; pipe 'y' to auto-confirm.
    try:
        subprocess.run(["rosnode", "cleanup"], input="y\n", text=True,
                       capture_output=True, timeout=5.0, check=False)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(1.0)  # let rosmaster fully drop stale publisher registrations

    print(f"[restart_planner] roslaunching exploration_multiagent.launch "
          f"(log: {_PLANNER_LOG_PATH}) ...")
    log_fh = open(_PLANNER_LOG_PATH, "w")
    _planner_proc = subprocess.Popen(
        ["roslaunch", "exploration_manager", "exploration_multiagent.launch"],
        stdout=log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    start = time.time()
    while time.time() - start < timeout:
        if _planner_proc.poll() is not None:
            raise RuntimeError(
                f"roslaunch exited prematurely (code {_planner_proc.returncode}). "
                f"See {_PLANNER_LOG_PATH}.")
        topics = [t for t, _ in rospy.get_published_topics()]
        if _PLANNER_PROBE_TOPIC in topics:
            print(f"[restart_planner] planner ready in {time.time()-start:.1f}s")
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Planner did not advertise {_PLANNER_PROBE_TOPIC} within {timeout}s. "
        f"Check {_PLANNER_LOG_PATH}.")


atexit.register(_kill_planner_proc)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    _kill_planner_proc()
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d", "hssd"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2, mp3d or hssd (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    # Optional episode filter (2026-05-19): if env var SKILLNAV_EPISODE_FILTER
    # points to a JSON list of {scene, epi} dicts (matching keys used in
    # videos/.../record.txt), only those episodes are run. Used to focus the
    # eval on FP-diff / StepOut-diff regression sets for fast iteration.
    episode_filter = None
    _filter_path = os.environ.get("SKILLNAV_EPISODE_FILTER", "").strip()
    if _filter_path:
        import json as _json
        try:
            with open(_filter_path) as _f:
                _raw = _json.load(_f)
            episode_filter = {(d["scene"], int(d["epi"])) for d in _raw}
            print(f"[EpisodeFilter] Loaded {len(episode_filter)} episodes from {_filter_path}")
        except Exception as _e:
            print(f"[EpisodeFilter] Failed to load {_filter_path}: {_e}; running unfiltered")
            episode_filter = None

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    itm_scores_multi_pub = rospy.Publisher(
        "/blip2_itm/scores", Float64MultiArray, queue_size=10
    )
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # If a regression filter is active, fast-forward past any episode that
        # is not in the requested subset. Skipped episodes do NOT touch the
        # planner restart, record.txt, or any aggregate counters.
        if episode_filter is not None:
            # scene_id is like ".../val/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb"
            # — the directory name (parent of the glb) is what record.txt logs.
            _scene_short = os.path.basename(os.path.dirname(env.current_episode.scene_id))
            _key = (_scene_short, int(env.current_episode.episode_id))
            while _key not in episode_filter:
                try:
                    env.current_episode = next(env.episode_iterator)
                except StopIteration:
                    print("[EpisodeFilter] Iterator exhausted; ending run.")
                    env.close()
                    pbar.close()
                    _kill_planner_proc()
                    return
                _scene_short = os.path.basename(os.path.dirname(env.current_episode.scene_id))
                _key = (_scene_short, int(env.current_episode.episode_id))
            print(f"[EpisodeFilter] Running filtered episode: {_key}")

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # HSSD ObjectNav uses underscored names (`potted_plant`, `tv`) for
        # `object_category`. Our LLM-answer file (llm_answer_hm3d.txt) and
        # downstream prompt logic expect HM3D-style spaces (`potted plant`).
        # Normalize the label for HSSD here; HM3D / MP3D already use spaces.
        label = label.replace("_", " ")

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Publish current target to ROS so planner can pick the matching prompt YAML.
        rospy.set_param("/target_object", str(label))
        rospy.set_param("/target_room", str(room))

        # Bind episode context to the call logger (one record per LLM/VLM call
        # written under this episode_id; see llm/instrumentation.py).
        _instr_set_episode(
            eid=env.current_episode.episode_id,
            scene=os.path.basename(env.current_episode.scene_id),
            target=str(label),
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Hard-reset the planner so it forgets the previous episode's
        # SDFMap / FrontierMap / VoronoiTopology state. /target_object and
        # /target_room are already set on the param server above and survive
        # the restart, so the new planner picks the correct prompt YAML.
        restart_planner()

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        # Per-episode wallclock guard: catches rare habitat-sim deadlocks so the
        # outer watchdog doesn't have to kill the whole config.
        _episode_wallclock_start = time.time()
        _EPISODE_WALLCLOCK_LIMIT_S = 360

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            if time.time() - _episode_wallclock_start > _EPISODE_WALLCLOCK_LIMIT_S:
                print(f"WARN: episode wallclock {_EPISODE_WALLCLOCK_LIMIT_S}s exceeded — force-ending")
                break
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)

            # Calculate ITM cosine similarities for the MultiValueMap fusion:
            #   sr_cosine -> semantic_relevance (target-direct) — drives ObjectMap
            #   ig_cosine -> information_gain  (corridor / doorway toward target)
            # Ablation: SKILLNAV_DISABLE_ITM=1 forces both to 0 (no semantic
            # signal; agent degrades to pure frontier exploration). Used in the
            # "−Exploration agent" ablation row.
            # SKILLNAV_DISABLE_SR=1 / _IG=1 zero only one channel.
            if os.environ.get("SKILLNAV_DISABLE_ITM"):
                sr_cosine, ig_cosine = 0.0, 0.0
            else:
                sr_cosine, ig_cosine = get_itm_message_cosines_dual(
                    observations["rgb"], label, room
                )
                if os.environ.get("SKILLNAV_DISABLE_SR"):
                    sr_cosine = 0.0
                if os.environ.get("SKILLNAV_DISABLE_IG"):
                    ig_cosine = 0.0
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity SR={sr_cosine:.3f} IG={ig_cosine:.3f}")

            # Legacy single-score channel (kept for backward compatibility; = SR)
            publish_float64(itm_score_pub, sr_cosine)
            # Two-score channel consumed by MultiValueMapManager
            multi_msg = Float64MultiArray()
            multi_msg.data = [float(sr_cosine), float(ig_cosine)]
            itm_scores_multi_pub.publish(multi_msg)

            # Detect objects in the current observation
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()
    _kill_planner_proc()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
