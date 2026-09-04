import numpy as np
import cv2
from collections import deque
from scipy.ndimage import binary_dilation

class VisualizationManager:
    def __init__(self, history_size=5, view_extent_m=10.0, render_resolution=0.05):
        self.history_size = history_size
        self.view_extent_m = float(view_extent_m)
        self.render_resolution = float(render_resolution)
        self.occupancy_history = deque(maxlen=history_size)  # Will store (grid, min_coords, robot_pose)
        self.resolution = 0.05  # 5cm per pixel
        self.inflation = 5      # inflation radius in pixels
    
    def reset(self):
        self.occupancy_history.clear()
        
    def build_occupancy_grid(self, depth_map, intrinsic, camera_roll=0):
        try:
            """Convert depth image to occupancy grid in BEV"""
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:,:,0]
            height, width = depth_map.shape
            uu, vv = np.meshgrid(np.arange(width), np.arange(height))
            z = np.asarray(depth_map, dtype=np.float32)
            # Mask invalid RTX pixels before projection. Multiplying image
            # coordinates by inf and filtering afterwards emits warnings and
            # can contaminate intermediate arrays with NaNs.
            valid_mask = (z > 0) & np.isfinite(z) & (z < 10)
            z_project = np.where(valid_mask, z, 0.0)
            x = (uu - intrinsic[0, 2]) * z_project / intrinsic[0, 0]
            y = (vv - intrinsic[1, 2]) * z_project / intrinsic[1, 1]

            # Filter valid points
            points_3d = np.stack((x[valid_mask], y[valid_mask], z[valid_mask]), axis=-1)
            
            # Apply camera roll
            roll = camera_roll * np.pi / 180
            rotation_matrix_x = np.array([[1, 0, 0], 
                                        [0, np.cos(roll), -np.sin(roll)], 
                                        [0, np.sin(roll), np.cos(roll)]])
            point_3d_flat = (rotation_matrix_x @ points_3d.transpose()).transpose()
            
            # Transform to world coordinates
            point_3d_world = np.zeros((point_3d_flat.shape[0], 3))
            point_3d_world[:, 0] = point_3d_flat[:, 2]
            point_3d_world[:, 1] = -point_3d_flat[:, 0]
            point_3d_world[:, 2] = -point_3d_flat[:, 1]
            bins = np.arange(np.min(point_3d_world[:, 2]), np.max(point_3d_world[:, 2]), 0.05)
            try:
                hist, bin_edges = np.histogram(point_3d_world[:, 2], bins=bins)
                max_freq_index = np.argmax(hist)
                point_3d_world[:, 2] -= bin_edges[max_freq_index]
                # print(f"bin_edges[max_freq_index] {bin_edges[max_freq_index]}")
            except:
                point_3d_world[:, 2] -= -0.5
            
            # Filter points within height range
            filtered_points = point_3d_world[(point_3d_world[:, 2] >= 0.2) & (point_3d_world[:, 2] <= 1.5)]
            if filtered_points.shape[0] == 0:
                min_coords = np.array([-5.0,-5.0,-5.0])
                max_coords = np.array([5.0,5.0,5.0])
                grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
                occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
                return occupancy_grid, min_coords
                
            # Create occupancy grid
            min_coords = np.min(filtered_points, axis=0)
            max_coords = np.max(filtered_points, axis=0)
            grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
            occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
            
            grid_coords = ((filtered_points[:, :2] - min_coords[:2]) / self.resolution).astype(int)
            occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]] = 1
            
        except:
            occupancy_grid = np.zeros((100,100),dtype=np.int8)
            min_coords = np.array([0,0])
        
        return occupancy_grid, min_coords
        
    def visualize_trajectory(self, rgb_image, depth_image, intrinsic, trajectory_points, robot_pose, camera_roll=0, all_trajectories_points=None, all_trajectories_values=None, all_trajectories_modes=None, selected_trajectory_index=None):
        # A tighter local view makes the 2-4 m candidate horizon readable.
        grid_size = int(self.view_extent_m / self.render_resolution)
        vis_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
         
        # Build current occupancy grid
        occupancy_grid, min_coords = self.build_occupancy_grid(depth_image[..., 0], intrinsic, camera_roll)
        if occupancy_grid is None:
            return combined_image
        
        # Add to history with robot pose
        self.occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        
        # Calculate center offset (assuming robot is at center)
        center_offset = grid_size // 2
        
        # Draw historical occupancy grids
        all_hist_world_points_list = []
        current_world_points = np.array([])

        # Process historical frames first
        for i, (hist_grid, hist_min_coords, hist_pose) in enumerate(self.occupancy_history):
            # Get occupied points in the grid's local frame
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            # Transform points from the grid's local frame to world frame
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]

            if i == len(self.occupancy_history) - 1:  # Current frame
                current_world_points = world_points
            else:  # Historical frame
                if world_points.size > 0:
                    all_hist_world_points_list.append(world_points)

        # Combine all historical points
        if all_hist_world_points_list:
            all_hist_world_points = np.concatenate(all_hist_world_points_list, axis=0)
        else:
            all_hist_world_points = np.array([])

        # Helper function to transform world points to vis_coords
        def transform_to_vis_coords(world_pts, current_pose, res, offset, size):
            if world_pts.size == 0:
                return np.array([])
                
            # Transform world points to yaw=0 frame centered at current robot position
            dx = world_pts[:, 0] - current_pose[0]
            dy = world_pts[:, 1] - current_pose[1]
            
            current_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates relative to center
            center_coords = (transformed_points / res).astype(int)
            
            # Filter points within visualization range
            valid_mask = (np.abs(center_coords[:, 0]) < size//2) & (np.abs(center_coords[:, 1]) < size//2)
            center_coords = center_coords[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_coords = np.zeros_like(center_coords)
            vis_coords[:, 0] = -center_coords[:, 0] + offset  # Flip x axis
            vis_coords[:, 1] = -center_coords[:, 1] + offset   # Keep y axis
            
            # Final boundary check
            valid_mask = (vis_coords[:, 0] >= 0) & (vis_coords[:, 0] < size) & \
                        (vis_coords[:, 1] >= 0) & (vis_coords[:, 1] < size)
            vis_coords = vis_coords[valid_mask]
            return vis_coords

        # Draw historical points (Gray)
        vis_coords_hist = transform_to_vis_coords(all_hist_world_points, robot_pose, self.render_resolution, center_offset, grid_size)
        if vis_coords_hist.size > 0:
            vis_image[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray

        # Draw current points (Red)
        vis_coords_current = transform_to_vis_coords(current_world_points, robot_pose, self.render_resolution, center_offset, grid_size)
        if vis_coords_current.size > 0:
            vis_image[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
        
        # Draw trajectory
        if trajectory_points is not None:
            # Transform trajectory points to yaw=0 frame centered at current robot position
            dx = trajectory_points[:, 0] - robot_pose[0]
            dy = trajectory_points[:, 1] - robot_pose[1]
            
            # Rotate points to align with yaw=0 frame
            current_rotation = np.array([
                [np.cos(0), np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.render_resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_points = np.zeros_like(grid_points)
            vis_points[:, 0] = -grid_points[:, 1] + center_offset  # Flip x axis
            vis_points[:, 1] = -grid_points[:, 0] + center_offset   # Keep y axis
            
            # Draw trajectory with anti-aliased lines
            for i in range(len(vis_points) - 1):
                cv2.line(vis_image, tuple(vis_points[i]), tuple(vis_points[i+1]), (0, 255, 0), 2, cv2.LINE_AA)
            # Draw start and end points
            if len(vis_points) > 0:
                # Use larger circles with anti-aliasing for smoother appearance
                cv2.circle(vis_image, tuple(vis_points[0]), 3, (255, 0, 0), -1, cv2.LINE_AA)  # Blue for start
                
                # Draw robot rectangle at start position
                rect_length = max(4, int(0.50 / self.render_resolution))
                rect_width = max(2, int(0.25 / self.render_resolution))
                start_point = (center_offset, center_offset)
                # Get yaw angle from trajectory points
                yaw = -robot_pose[2]
                
                # Calculate rectangle corners
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                corners = np.array([
                    [-rect_width/2, -rect_length/2],
                    [rect_width/2, -rect_length/2],
                    [rect_width/2, rect_length/2],
                    [-rect_width/2, rect_length/2]
                ])
                # Rotate corners
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = (rot_matrix @ corners.T).T + start_point
                
                # Draw rectangle with anti-aliasing
                corners_int = rotated_corners.astype(np.int32)
                cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # Blue rectangle
                cv2.circle(vis_image, tuple(vis_points[-1]), 3, (0, 0, 255), -1, cv2.LINE_AA)  # Red for end
        
        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
        
        # If no all_trajectories_points, return original combined image
        if all_trajectories_points is None or len(all_trajectories_points) == 0:
            return combined_image
        # print(f"all_trajectories_points: {len(all_trajectories_points)}")
        # --- Create additional visualization for all trajectories ---
        # Create a new image for all trajectories visualization
        vis_image_all = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # Draw the same occupancy grid
        if vis_coords_hist.size > 0:
            vis_image_all[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray
        if vis_coords_current.size > 0:
            vis_image_all[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
            
        # Draw all trajectories with colors based on values
        # Define color mapping function from value to color (blue to red gradient)
        def value_to_color(value, values_min, values_max):
            # Normalize value to [0, 1] based on a fixed range [-2, 0.5]
            fixed_min = -1.2
            fixed_max = 0.2

            # Clamp the value to be within the fixed range
            value = np.clip(value, fixed_min, fixed_max)

            # Normalize value to [0, 1]
            normalized = (value - fixed_min) / (fixed_max - fixed_min)

            # Map to blue (low) -> green (mid) -> red (high)
            if normalized < 0.5:
                # Blue to green
                b = 255 * (1 - 2 * normalized)
                g = 255 * (2 * normalized)
                r = 0
            else:
                # Green to red
                b = 0
                g = 255 * (2 - 2 * normalized)
                r = 255 * (2 * normalized - 1)
            
            return (int(b), int(g), int(r))  # Return BGR color
        
        # Fixed RGB palette; yellow is reserved exclusively for selection.
        mode_colors = (
            (50, 120, 255),   # m0: blue
            (0, 220, 255),    # m1: cyan
            (60, 255, 80),    # m2: green
            (255, 80, 50),    # m3: red
            (230, 80, 255),   # m4: magenta
        )
        if all_trajectories_modes is not None:
            # Official NavDP returns candidates without mode-debug metadata.
            # A short/empty debug list must not shorten the color table below
            # the actual number of trajectories.
            trajectory_colors = []
            for idx in range(len(all_trajectories_points)):
                mode = (
                    int(all_trajectories_modes[idx])
                    if idx < len(all_trajectories_modes) else idx
                )
                trajectory_colors.append(mode_colors[mode % len(mode_colors)])
        elif all_trajectories_values is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            trajectory_colors = [colors[idx % len(colors)] for idx in range(len(all_trajectories_points))]
        else:
            # Get min and max values for normalization
            values_min = np.min(all_trajectories_values)
            values_max = np.max(all_trajectories_values)
            # Generate color for each trajectory
            trajectory_colors = [value_to_color(v, values_min, values_max) for v in all_trajectories_values]
        
        label_specs = []
        draw_order = list(range(len(all_trajectories_points)))
        if selected_trajectory_index is not None:
            selected_index = int(selected_trajectory_index)
            if selected_index in draw_order:
                draw_order.remove(selected_index)
                draw_order.append(selected_index)
        selected_color = (255, 255, 0)  # RGB bright yellow in imageio output.
        for idx in draw_order:
            traj = all_trajectories_points[idx]
            color = trajectory_colors[idx]
            
            # Transform trajectory points
            dx = traj[:, 0] - robot_pose[0]
            dy = traj[:, 1] - robot_pose[1]
            
            # Rotate points
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.render_resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates
            vis_points_all = np.zeros_like(grid_points)
            vis_points_all[:, 0] = -grid_points[:, 1] + center_offset
            vis_points_all[:, 1] = -grid_points[:, 0] + center_offset
            
            # Draw trajectory with anti-aliased lines
            selected = selected_trajectory_index is not None and idx == int(selected_trajectory_index)
            draw_color = selected_color if selected else color
            thickness = 4 if selected else 1
            for i in range(len(vis_points_all) - 1):
                cv2.line(vis_image_all, tuple(vis_points_all[i]), tuple(vis_points_all[i+1]), draw_color, thickness, cv2.LINE_AA)
                
            # Draw start and end points with anti-aliasing
            if len(vis_points_all) > 0:
                cv2.circle(vis_image_all, tuple(vis_points_all[0]), 2, draw_color, -1, cv2.LINE_AA)
                if all_trajectories_modes is not None and idx < len(all_trajectories_modes):
                    anchor_index = min(
                        len(vis_points_all) - 1,
                        max(0, int((0.52 + 0.10 * (idx % 4)) * len(vis_points_all))),
                    )
                    label_specs.append((
                        vis_points_all[anchor_index].copy(),
                        "m%d" % int(all_trajectories_modes[idx]),
                        draw_color,
                    ))
        
        # Draw robot position with anti-aliasing
        if len(vis_points) > 0:
            corners_int = rotated_corners.astype(np.int32)
            cv2.polylines(vis_image_all, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # White robot outline
        
        # First-person and candidate map occupy equal-width halves. Pad the RGB
        # vertically instead of distorting the square metric map.
        target_width = rgb_image.shape[1]
        target_height = target_width
        vis_resized_all = cv2.resize(
            vis_image_all, (target_width, target_height),
            interpolation=cv2.INTER_AREA if grid_size > target_width else cv2.INTER_LINEAR,
        )
        # Draw labels after resizing so their font stays readable and does not
        # get magnified with the metric grid. Greedy placement avoids overlap.
        occupied_label_boxes = []
        label_scale = target_width / float(grid_size)
        label_offsets = (
            (10, -8), (10, 20), (-42, -8), (-42, 20),
            (28, -28), (28, 38), (-60, -28), (-60, 38),
            (48, -48), (48, 58), (-80, -48), (-80, 58),
            (10, -68), (10, 78), (-42, -68), (-42, 78),
        )
        for label_index, (anchor_grid, mode_text, color) in enumerate(label_specs):
            anchor = np.rint(anchor_grid * label_scale).astype(int)
            text_size, baseline = cv2.getTextSize(
                mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
            )
            label_box = None
            label_origin = None
            ordered_offsets = label_offsets[label_index % len(label_offsets):] + label_offsets[:label_index % len(label_offsets)]
            for offset_x, offset_y in ordered_offsets:
                x = int(np.clip(anchor[0] + offset_x, 2, target_width - text_size[0] - 4))
                y = int(np.clip(anchor[1] + offset_y, text_size[1] + 4, target_height - baseline - 3))
                box = (x - 3, y - text_size[1] - 3,
                       x + text_size[0] + 3, y + baseline + 3)
                overlaps = any(
                    not (box[2] < old[0] or box[0] > old[2]
                         or box[3] < old[1] or box[1] > old[3])
                    for old in occupied_label_boxes
                )
                if not overlaps:
                    label_box, label_origin = box, (x, y)
                    break
            if label_box is None:
                continue
            occupied_label_boxes.append(label_box)
            box_center = ((label_box[0] + label_box[2]) // 2,
                          (label_box[1] + label_box[3]) // 2)
            cv2.line(vis_resized_all, tuple(anchor), box_center, color, 1, cv2.LINE_AA)
            cv2.rectangle(vis_resized_all, label_box[:2], label_box[2:],
                          (15, 15, 15), -1)
            cv2.putText(vis_resized_all, mode_text, label_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        rgb_panel = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        rgb_y = max((target_height - rgb_image.shape[0]) // 2, 0)
        copy_height = min(rgb_image.shape[0], target_height)
        rgb_panel[rgb_y:rgb_y + copy_height] = rgb_image[:copy_height]
        final_combined_image = np.concatenate((rgb_panel, vis_resized_all), axis=1)
        
        return final_combined_image
