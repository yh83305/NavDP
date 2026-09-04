import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./assets/scenes/cluttered_hard")
parser.add_argument(
    "--scene_index", type=int, default=8)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--speed", type=float, default=0.5)
parser.add_argument(
    "--port", type=int, default=8888)
parser.add_argument("--perf_steps", type=int, default=0,
                    help="Exit after this many measured control-loop steps.")
parser.add_argument("--perf_warmup", type=int, default=20,
                    help="Control-loop steps excluded from timing statistics.")
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading
import json
from datetime import datetime

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_health,navigator_reset,pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller

planning_input = PlanningInput() 
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
perf_lock = threading.Lock()
perf_samples = {name: [] for name in (
    "loop", "observation_readback", "visualization", "mpc", "env_step",
    "planning",
)}
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mode_vis_manager = [
    VisualizationManager(history_size=5, view_extent_m=6.0, render_resolution=0.01)
    for i in range(args_cli.num_envs)
]
mpc = None

# Only our explicit-mode FLUX server implements the candidate mode/debug
# contract. Every official or third-party NavDP benchmark method keeps the
# original value-colored visualization even if it returns extra diagnostics.
MODE_DEBUG_VISUALIZATION_ALGOS = {
    "flux_explicit_modes_rule16",
    "flux_direction5_speed3_rule16",
}

def planning_thread(env, camera_intrinsic):
    global mpc
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
                observation_time = (
                    None if planning_input.observation_time is None
                    else planning_input.observation_time.copy()
                )
            with output_lock:
                planning_output.is_planning = True
            
            # Start timing planning
            planning_start = time.time()
            result = pointgoal_step(
                goal, image, depth, port=args_cli.port,
                observation_time=observation_time,
            )
            trajectory_points_camera, all_trajectories_camera, all_values_camera = result[:3]
            mode_debug = result[3] if len(result) >= 4 and isinstance(result[3], list) and result[3] and isinstance(result[3][0], dict) and "candidate_debug" in result[3][0] else None
            # Transform trajectory from camera frame to world frame
            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                mpc = MPC_Controller(trajectory_points_world,
                                     desired_v=args_cli.speed,
                                     v_max=args_cli.speed,
                                     w_max=args_cli.speed)
            batch_optimal_points_world = np.array(batch_optimal_points_world)
           
            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                # Transform all trajectories
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            # Update shared state
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_trajectories_camera = all_trajectories_camera.copy()
                planning_output.point_goals_camera = goal.copy()
                planning_output.all_values_camera = all_values_camera
                planning_output.mode_debug = mode_debug
                planning_output.is_planning = False
                planning_output.planning_error = None
            
            # Print planning timing
            planning_time = time.time() - planning_start
            with perf_lock:
                perf_samples["planning"].append(planning_time)
            # print(f"Planning time: {planning_time:.3f}s, Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

def _format_debug_number(value, *, signed=False):
    """Format nullable/non-finite JSON diagnostics without crashing video."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def draw_mode_debug_panel(image, debug):
    """Append a readable candidate decision table to a benchmark video frame."""
    if not debug or not isinstance(debug, list):
        return image
    item = debug[0] if isinstance(debug[0], dict) else {}
    rows = item.get("candidate_debug", [])
    panel_width = 620
    map_height = 0
    panel_height = max(image.shape[0], map_height + 40 + 24 * (len(rows) + 3))
    panel = np.full((panel_height, panel_width, 3), (18, 22, 28), dtype=np.uint8)
    cv2.putText(panel, f"MODE SELECT DEBUG  reason={item.get('selection_reason', 'unknown')}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)
    table_top = map_height + 18
    # Hershey fonts are proportional; fixed x positions keep every column aligned.
    columns = ((8, "idx"), (45, "mode"), (92, "P"), (132, "safe"),
               (174, "ent"), (213, "goal"), (281, "esdf"), (351, "unk"),
               (395, "temp"), (458, "final"), (530, "filter"))
    for x, label in columns:
        cv2.putText(panel, label, (x, table_top + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (190, 200, 210), 1, cv2.LINE_AA)
    for line, candidate in enumerate(rows, start=2):
        selected = bool(candidate.get("selected"))
        safe = bool(candidate.get("safe"))
        if selected:
            color = (20, 220, 255)
        elif safe:
            color = (80, 220, 100)
        else:
            color = (100, 110, 125)
        y = table_top + 27 + line * 24
        values = (
            (8, f"{candidate.get('index', -1):02d}"),
            (45, f"m{candidate.get('mode', -1)}"),
            (92, _format_debug_number(candidate.get("prior"))),
            (132, "Y" if safe else "N"),
            (174, "Y" if candidate.get("entered_selection") else "N"),
            (213, _format_debug_number(candidate.get("goal_score"), signed=True)),
            (281, _format_debug_number(candidate.get("minimum_esdf_clearance_m"), signed=True)),
            (351, _format_debug_number(candidate.get("unknown_fraction"))),
            (395, _format_debug_number(candidate.get("temporal_cost"))),
            (458, _format_debug_number(candidate.get("final_score"), signed=True)),
            (530, str(candidate.get('filtered_reason') or ('SELECTED' if selected else '-'))),
        )
        for x, value in values:
            cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, color, 1, cv2.LINE_AA)
    if image.shape[0] < panel_height:
        image = np.pad(image, ((0, panel_height - image.shape[0]), (0, 0), (0, 0)))
    return np.concatenate((image, panel), axis=1)


def draw_esdf_trajectory_overlay(size, trajectories_camera, point_goal, debug):
    """Draw camera-local candidates directly over the metric ESDF slice."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if not debug or not debug.get("esdf_debug"):
        return canvas
    metadata = debug["esdf_debug"]
    esdf = np.asarray(metadata.get("slice", []), dtype=np.uint8)
    if esdf.ndim != 2 or not esdf.size:
        return canvas
    # applyColorMap returns BGR while the benchmark video buffer is RGB.
    esdf_rgb = cv2.cvtColor(
        cv2.applyColorMap(255 - esdf, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    scale = min(size / float(esdf.shape[0]), size / float(esdf.shape[1]))
    map_height = max(1, int(round(esdf.shape[0] * scale)))
    map_width = max(1, int(round(esdf.shape[1] * scale)))
    map_rgb = cv2.resize(
        esdf_rgb, (map_width, map_height), interpolation=cv2.INTER_NEAREST
    )
    offset_x = (size - map_width) // 2
    offset_y = (size - map_height) // 2
    canvas[offset_y:offset_y + map_height, offset_x:offset_x + map_width] = map_rgb

    rows = debug.get("candidate_debug", [])
    modes = [int(row.get("mode", -1)) for row in rows]
    selected_index = int(debug.get("selected_index", -1))
    mode_colors = (
        (50, 120, 255), (0, 220, 255), (60, 255, 80),
        (255, 80, 50), (230, 80, 255),
    )
    trajectories = np.asarray(trajectories_camera, dtype=np.float32)
    draw_order = list(range(len(trajectories)))
    if selected_index in draw_order:
        draw_order.remove(selected_index)
        draw_order.append(selected_index)
    origin_right, origin_forward = metadata.get(
        "grid_origin_right_forward_m", [-2.0, 0.0]
    )
    voxel = max(float(metadata.get("voxel_size_m", 0.05)), 1e-6)
    occupied_labels = []
    for index in draw_order:
        trajectory = trajectories[index]
        right = -trajectory[:, 1]
        forward = trajectory[:, 0]
        column = (right - float(origin_right)) / voxel
        original_row = (forward - float(origin_forward)) / voxel
        display_row = (esdf.shape[0] - 1) - original_row
        pixels = np.rint(np.stack((
            offset_x + column * scale,
            offset_y + display_row * scale,
        ), axis=1)).astype(np.int32)
        valid = (
            (pixels[:, 0] >= offset_x)
            & (pixels[:, 0] < offset_x + map_width)
            & (pixels[:, 1] >= offset_y)
            & (pixels[:, 1] < offset_y + map_height)
        )
        pixels = pixels[valid]
        if len(pixels) < 2:
            continue
        selected = index == selected_index
        mode = modes[index] if index < len(modes) else -1
        color = (255, 255, 0) if selected else mode_colors[mode % 5]
        cv2.polylines(
            canvas, [pixels], False, color, 4 if selected else 2, cv2.LINE_AA
        )
        anchor = pixels[min(len(pixels) - 1, int(0.70 * len(pixels)))]
        label = f"m{mode}"
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        offsets = ((6, -6), (6, 18), (-38, -6), (-38, 18),
                   (22, -28), (-54, -28), (22, 38), (-54, 38))
        placed = None
        for dx, dy in offsets[index % len(offsets):] + offsets[:index % len(offsets)]:
            x = int(np.clip(anchor[0] + dx, 2, size - text_size[0] - 4))
            y = int(np.clip(anchor[1] + dy, text_size[1] + 4, size - baseline - 3))
            box = (x - 3, y - text_size[1] - 3,
                   x + text_size[0] + 3, y + baseline + 3)
            if not any(
                not (box[2] < old[0] or box[0] > old[2]
                     or box[3] < old[1] or box[1] > old[3])
                for old in occupied_labels
            ):
                placed = (box, (x, y))
                break
        if placed is not None:
            box, text_origin = placed
            occupied_labels.append(box)
            cv2.rectangle(canvas, box[:2], box[2:], (12, 12, 12), -1)
            cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

    goal = np.asarray(point_goal, dtype=np.float32).reshape(-1)
    if len(goal) >= 2 and bool(np.isfinite(goal[:2]).all()):
        goal_right = -float(goal[1])
        goal_forward = float(goal[0])
        goal_column = (goal_right - float(origin_right)) / voxel
        goal_original_row = (goal_forward - float(origin_forward)) / voxel
        goal_display_row = (esdf.shape[0] - 1) - goal_original_row
        raw_goal = np.array([
            offset_x + goal_column * scale,
            offset_y + goal_display_row * scale,
        ], dtype=np.float64)
        margin = 14
        goal_pixel = np.rint(np.clip(
            raw_goal,
            [offset_x + margin, offset_y + margin],
            [offset_x + map_width - margin, offset_y + map_height - margin],
        )).astype(np.int32)
        inside = (
            offset_x <= raw_goal[0] < offset_x + map_width
            and offset_y <= raw_goal[1] < offset_y + map_height
        )
        cv2.circle(canvas, tuple(goal_pixel), 11, (255, 255, 255), 2,
                   cv2.LINE_AA)
        cv2.drawMarker(canvas, tuple(goal_pixel), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        goal_distance = float(np.linalg.norm(goal[:2]))
        goal_label = f"GOAL {goal_distance:.1f}m" + ("" if inside else " >")
        text_x = int(np.clip(goal_pixel[0] + 14, 2, size - 105))
        text_y = int(np.clip(goal_pixel[1] - 10, 18, size - 4))
        cv2.putText(canvas, goal_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.putText(canvas, "ESDF + candidates", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "red=occupied  blue=clear", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    return canvas

scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
usd_path,init_path = find_usd_path(scene_path,task='pointgoal')
scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal = GOAL_CFG
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path":init_path, 
                                       'height_offset':0.1,
                                       'robot_visible': False,
                                       'light_enabled': False}
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
_,infos = env.reset()
# warm-up
PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
    
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control", 
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)
run_started_at = datetime.now().astimezone()
try:
    server_metadata = navigator_health(port=args_cli.port)
except Exception as error:
    server_metadata = {"health_error": repr(error)}

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
run_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
save_dir = "./pointgoal_%s_%s/%s_%s/"%(
    algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2],run_timestamp
)
os.makedirs(save_dir,exist_ok=True)
run_metadata = {
    "started_at": run_started_at.isoformat(),
    "algorithm": algo,
    "scene_dir": os.path.abspath(args_cli.scene_dir),
    "scene_index": args_cli.scene_index,
    "scene_path": scene_path,
    "num_envs": args_cli.num_envs,
    "num_episodes": args_cli.num_episodes,
    "speed": args_cli.speed,
    "stop_threshold": args_cli.stop_threshold,
    "server_port": args_cli.port,
    "server": server_metadata,
}
with open(os.path.join(save_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
    json.dump(run_metadata, file, ensure_ascii=False, indent=2)
print("[NavDP Eval] output=%s" % os.path.abspath(save_dir))

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]

trajectory_length = np.zeros((scene_config.num_envs))
perf_loop_index = 0

def record_perf(name, elapsed):
    if perf_loop_index >= args_cli.perf_warmup:
        with perf_lock:
            perf_samples[name].append(float(elapsed))

def print_perf_summary():
    with perf_lock:
        samples = {name: np.asarray(values, dtype=np.float64)
                   for name, values in perf_samples.items() if values}
    print("[NavDP PERF] nominal_physics_hz=%.3f nominal_control_hz=%.3f step_dt=%.3fs" % (
        1.0 / env.unwrapped.cfg.sim.dt,
        1.0 / env.unwrapped.step_dt,
        env.unwrapped.step_dt,
    ))
    for name, values in samples.items():
        print("[NavDP PERF] %s count=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f" % (
            name, len(values), 1000.0 * values.mean(),
            1000.0 * np.percentile(values, 50),
            1000.0 * np.percentile(values, 95),
        ))
    loops = samples.get("loop")
    if loops is not None and len(loops):
        print("[NavDP PERF] actual_control_hz=%.3f realtime_factor=%.3f" % (
            1.0 / loops.mean(), env.unwrapped.step_dt / loops.mean()
        ))

while simulation_app.is_running():
    loop_start = time.perf_counter()
    with torch.inference_mode():
        stage_start = time.perf_counter()
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        record_perf("observation_readback", time.perf_counter() - stage_start)
        
        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()
            planning_input.observation_time = (
                env.unwrapped.episode_length_buf.float().cpu().numpy()
                * float(env.unwrapped.step_dt)
            )

        # based on the current world trajectory 
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

        x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
        current_trajectory = None
        current_all_trajectories = None
        current_all_trajectories_camera = None
        current_point_goals_camera = None
        current_all_values = None
        current_mode_debug = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_trajectories_camera = planning_output.all_trajectories_camera.copy() if planning_output.all_trajectories_camera is not None else None
                current_point_goals_camera = planning_output.point_goals_camera.copy() if planning_output.point_goals_camera is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
                current_mode_debug = planning_output.mode_debug
        
        if current_trajectory is not None:
            control_start = time.time()
            action_list = []
            for i in range(args_cli.num_envs):
                stage_start = time.perf_counter()
                # Keep the official NavDP visualization independent from the
                # explicit-mode diagnostics returned by our FLUX server.
                # NavDP has no mode metadata and should retain its original
                # value-colored candidate rendering.
                use_mode_visualization = (
                    algo in MODE_DEBUG_VISUALIZATION_ALGOS
                    and current_mode_debug is not None
                    and i < len(current_mode_debug)
                )
                candidate_rows = (
                    current_mode_debug[i].get("candidate_debug", [])
                    if use_mode_visualization else None
                )
                active_vis_manager = mode_vis_manager[i] if use_mode_visualization else vis_manager[i]
                vis_image = active_vis_manager.visualize_trajectory(
                    images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i],
                    all_trajectories_modes=(
                        [row.get("mode", -1) for row in candidate_rows]
                        if use_mode_visualization else None
                    ),
                    selected_trajectory_index=(
                        current_mode_debug[i].get("selected_index")
                        if use_mode_visualization else None
                    ),
                )
                if (use_mode_visualization
                        and current_all_trajectories_camera is not None
                        and current_point_goals_camera is not None):
                    overlay = draw_esdf_trajectory_overlay(
                        vis_image.shape[0],
                        current_all_trajectories_camera[i],
                        current_point_goals_camera[i],
                        current_mode_debug[i],
                    )
                    vis_image[:, -vis_image.shape[0]:] = overlay
                    vis_image = draw_mode_debug_panel(vis_image, [current_mode_debug[i]])
                record_perf("visualization", time.perf_counter() - stage_start)
                if mpc is None:
                    continue
                t0 = time.time()
                opt_u_controls, opt_x_states = mpc.solve(x0[i,:3])
                record_perf("mpc", time.time() - t0)
                print(f"solve mpc cost {time.time() - t0}")
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)
                
                try:
                    vis_image = draw_box_with_text(vis_image,0,0,430,50,"desired lin.:%.2f ang.:%.2f"%(v,w))
                    vis_image = draw_box_with_text(vis_image,0,50,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel,robot_ang_vel))
                    if current_all_values is not None:
                        vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values[i]), np.min(current_all_values[i])))
                    vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
                
            action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
            stage_start = time.perf_counter()
            obs, rewards, dones, infos = env.step(action)
            record_perf("env_step", time.perf_counter() - stage_start)
            # Get actual joint velocities from Isaac Sim
            actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
            desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
            trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
        else:
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            stage_start = time.perf_counter()
            obs, rewards, dones, infos = env.step(action)
            record_perf("env_step", time.perf_counter() - stage_start)
            print("No trajectory available, using zero action")
        
        for i in range(args_cli.num_envs):
            if dones[i] == True:
                episode_num += 1
                navigator_reset(env_id=i,port=args_cli.port)
                success_flag = (np.sqrt(np.square(goals[i]).sum())<1.5).astype(np.float32)
                fps_writer[i].close()
                evaluation_metrics.append({'success':success_flag,
                                           'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                           'distance':euclidean[i]})
                write_metrics(evaluation_metrics,save_dir+"metric.csv")
                euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0
        
        if episode_num >= args_cli.num_episodes:
            break
        record_perf("loop", time.perf_counter() - loop_start)
        perf_loop_index += 1
        if args_cli.perf_steps > 0 and perf_loop_index >= (
            args_cli.perf_warmup + args_cli.perf_steps
        ):
            print_perf_summary()
            break
       
                
   

        
