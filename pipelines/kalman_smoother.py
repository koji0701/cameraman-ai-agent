import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, Dict
import av
from pathlib import Path
import json


def interpolate_and_smooth_coordinates(
    normalized_crops: pd.DataFrame,
    video_path: str = None,
    target_fps: float = None,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    enable_debug_outputs: bool = False,
    debug_outputs_dir: str = None
) -> pd.DataFrame:
    """Complete pipeline: Interpolate crop coordinates to per-frame and apply Kalman smoothing."""
    
    if enable_debug_outputs and debug_outputs_dir:
        interpolation_config = {
            "input_keyframes": len(normalized_crops),
            "smoothing_strength": smoothing_strength,
            "interpolation_method": interpolation_method,
            "target_fps": target_fps,
            "video_path": video_path
        }
        save_kalman_step_data(interpolation_config, "03_interpolation_config", debug_outputs_dir, "Kalman filter configuration")
    
    fps = target_fps
    if fps is None and video_path:
        fps = detect_video_fps(video_path)
        print(f"Detected video FPS: {fps}")
    if fps is None:
        fps = 30.0
        print(f"Using default FPS: {fps}")
    
    print(f"Interpolating coordinates from {len(normalized_crops)} keyframes to {fps} FPS...")
    interpolated_df = interpolate_coordinates_to_frames(normalized_crops, fps, method=interpolation_method)
    
    if enable_debug_outputs and debug_outputs_dir:
        save_kalman_step_data(interpolated_df, "03_interpolated_coordinates", debug_outputs_dir, f"Interpolated to {len(interpolated_df)} frames")
    
    print(f"Applying Kalman smoothing (strength: {smoothing_strength})...")
    smoothed_df = apply_kalman_smoothing(
        interpolated_df,
        fps,
        smoothing_strength=smoothing_strength,
        enable_debug_outputs=enable_debug_outputs,
        debug_outputs_dir=debug_outputs_dir
    )
    
    print(f"Final validation and constraint checking...")
    final_df = validate_and_constrain_coordinates(smoothed_df)
    
    if enable_debug_outputs and debug_outputs_dir:
        save_kalman_step_data(final_df, "03_final_smoothed_coordinates", debug_outputs_dir, f"Final validated coordinates for {len(final_df)} frames")
        
        before_after_comparison = {
            "original_keyframes": len(normalized_crops),
            "interpolated_frames": len(interpolated_df),
            "final_smooth_frames": len(final_df),
            "fps_used": fps,
            "duration_seconds": len(final_df) / fps,
            "smoothing_applied": smoothing_strength,
            "interpolation_method": interpolation_method
        }
        save_kalman_step_data(before_after_comparison, "03_smoothing_summary", debug_outputs_dir, "Complete smoothing pipeline summary")
    
    print(f"✓ Kalman smoothing complete: {len(normalized_crops)} → {len(final_df)} frames")
    return final_df


def detect_video_fps(video_path: str) -> float:
    """Detect the FPS of a video file using PyAV."""
    try:
        with av.open(video_path) as container:
            fps = float(container.streams.video[0].average_rate)
            return fps
    except Exception as e:
        print(f"Warning: Could not detect FPS ({e}), using default 30.0")
        return 30.0


def interpolate_coordinates_to_frames(
    normalized_crops: pd.DataFrame, 
    fps: float,
    method: str = 'cubic'
) -> pd.DataFrame:
    """Interpolate sparse keyframe coordinates to per-frame coordinates."""
    
    if len(normalized_crops) < 2:
        raise ValueError("Need at least 2 keyframes for interpolation")
    
    keyframe_times = normalized_crops['t_ms'].values / 1000.0
    max_time = keyframe_times[-1]
    
    frame_times = np.arange(0, max_time + 1/fps, 1/fps)
    
    coord_columns = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    interpolated_coords = {}
    
    for col in coord_columns:
        values = normalized_crops[col].values
        
        if method == 'cubic' and len(normalized_crops) >= 4:
            spline = CubicSpline(keyframe_times, values, bc_type='natural')
            interpolated_coords[col] = spline(frame_times)
        elif method == 'quadratic' and len(normalized_crops) >= 3:
            from scipy.interpolate import interp1d
            interp_func = interp1d(keyframe_times, values, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            interpolated_coords[col] = interp_func(frame_times)
        else:
            from scipy.interpolate import interp1d
            interp_func = interp1d(keyframe_times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated_coords[col] = interp_func(frame_times)
    
    result_df = pd.DataFrame({
        't_ms': frame_times * 1000,
        'frame_number': np.arange(len(frame_times)),
        **interpolated_coords
    })
    
    return result_df


def save_kalman_step_data(data, step_name: str, output_dir: str, description: str = ""):
    """Save Kalman filter step data to debug outputs."""
    if not output_dir:
        return
        
    output_path = Path(output_dir) / f"step_{step_name}.csv" if isinstance(data, pd.DataFrame) else Path(output_dir) / f"step_{step_name}.json"
    
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        print(f"🔍 KALMAN DEBUG: Saved {step_name} → {output_path}")
        if description:
            print(f"                  {description}")
            
    except Exception as e:
        print(f"⚠️ Could not save {step_name}: {e}")


def apply_kalman_smoothing(
    interpolated_df: pd.DataFrame,
    fps: float,
    smoothing_strength: str = 'balanced',
    enable_debug_outputs: bool = False,
    debug_outputs_dir: str = None
) -> pd.DataFrame:
    """Apply Kalman filtering to smooth interpolated coordinates."""
    
    strength_configs = {
        'minimal': {'process_noise': 0.1, 'measurement_noise': 1.0},
        'balanced': {'process_noise': 0.05, 'measurement_noise': 0.5},
        'maximum': {'process_noise': 0.01, 'measurement_noise': 0.1},
        'cinematic': {'process_noise': 0.005, 'measurement_noise': 0.05}
    }
    
    config = strength_configs.get(smoothing_strength, strength_configs['balanced'])
    dt = 1.0 / fps
    
    if enable_debug_outputs and debug_outputs_dir:
        kalman_config = {
            "smoothing_strength": smoothing_strength,
            "process_noise": config['process_noise'],
            "measurement_noise": config['measurement_noise'],
            "dt": dt,
            "fps": fps,
            "total_frames": len(interpolated_df)
        }
        save_kalman_step_data(kalman_config, "03_kalman_filter_config", debug_outputs_dir, f"Kalman filter configuration for {smoothing_strength} smoothing")
    
    coord_columns = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    smoothed_coords = np.zeros((len(interpolated_df), len(coord_columns)))
    
    kalman_details = {}
    
    for i, col in enumerate(coord_columns):
        measurements = interpolated_df[col].values
        
        kf = create_position_velocity_kalman_filter(
            dt=dt,
            process_noise=config['process_noise'],
            measurement_noise=config['measurement_noise']
        )
        
        kf.x = np.array([measurements[0], 0.0])
        
        states = []
        covariances = []
        
        for measurement in measurements:
            kf.predict()
            kf.update(measurement)
            
            states.append(kf.x.copy())
            covariances.append(kf.P.copy())
            
            smoothed_coords[i if i < len(measurements) else len(measurements)-1, i] = kf.x[0]
        
        kalman_details[col] = {
            "initial_measurement": float(measurements[0]),
            "final_measurement": float(measurements[-1]),
            "initial_smoothed": float(smoothed_coords[0, i]),
            "final_smoothed": float(smoothed_coords[-1, i]),
            "variance_reduction": float(np.var(measurements) - np.var(smoothed_coords[:, i])),
            "total_frames_processed": len(measurements)
        }
    
    if enable_debug_outputs and debug_outputs_dir:
        save_kalman_step_data(kalman_details, "03_kalman_filter_details", debug_outputs_dir, "Detailed Kalman filter results per coordinate")
    
    result_df = interpolated_df.copy()
    result_df['crop_x'] = smoothed_coords[:, 0]
    result_df['crop_y'] = smoothed_coords[:, 1]
    result_df['crop_w'] = smoothed_coords[:, 2]
    result_df['crop_h'] = smoothed_coords[:, 3]
    
    return result_df


def create_position_velocity_kalman_filter(
    dt: float,
    process_noise: float,
    measurement_noise: float
) -> KalmanFilter:
    """Create a Kalman filter for position-velocity tracking."""
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    kf.F = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])
    
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise)
    kf.R = np.array([[measurement_noise]])
    kf.P *= 100.0
    
    return kf


def validate_and_constrain_coordinates(
    smoothed_df: pd.DataFrame,
    frame_width: int = 1920,
    frame_height: int = 1080,
    min_crop_size: int = 100
) -> pd.DataFrame:
    """Validate and constrain smoothed coordinates to ensure valid video crops."""
    
    result_df = smoothed_df.copy()
    coord_cols = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    
    for col in coord_cols:
        mask = ~np.isfinite(result_df[col])
        if mask.any():
            print(f"Warning: Found {mask.sum()} non-finite values in {col}, interpolating...")
            result_df[col] = result_df[col].interpolate(method='linear')
            result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
            
            if col in ['crop_x', 'crop_y']:
                result_df[col] = result_df[col].fillna(0)
            else:
                result_df[col] = result_df[col].fillna(min_crop_size)
    
    result_df[coord_cols] = result_df[coord_cols].round().astype(int)
    
    result_df['crop_x'] = np.clip(result_df['crop_x'], 0, frame_width)
    result_df['crop_y'] = np.clip(result_df['crop_y'], 0, frame_height)
    
    result_df['crop_w'] = np.maximum(result_df['crop_w'], min_crop_size)
    result_df['crop_h'] = np.maximum(result_df['crop_h'], min_crop_size)
    
    result_df['crop_w'] = np.minimum(result_df['crop_w'], frame_width - result_df['crop_x'])
    result_df['crop_h'] = np.minimum(result_df['crop_h'], frame_height - result_df['crop_y'])
    
    result_df['crop_w'] = result_df['crop_w'] - (result_df['crop_w'] % 2)
    result_df['crop_h'] = result_df['crop_h'] - (result_df['crop_h'] % 2)
    
    result_df['zoom_factor'] = np.minimum(
        frame_width / result_df['crop_w'],
        frame_height / result_df['crop_h']
    )
    
    return result_df


def analyze_motion_smoothness(smoothed_df: pd.DataFrame, fps: float) -> Dict:
    """Analyze the smoothness and quality of the motion after Kalman filtering."""
    
    if len(smoothed_df) < 3:
        return {"error": "Insufficient data for motion analysis"}
    
    dt = 1.0 / fps
    coords = smoothed_df[['crop_x', 'crop_y', 'crop_w', 'crop_h']].values
    velocities = np.diff(coords, axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt
    
    speed = np.sqrt(np.sum(velocities[:, :2]**2, axis=1))
    acceleration_magnitude = np.sqrt(np.sum(accelerations[:, :2]**2, axis=1))
    
    zoom_factors = smoothed_df['zoom_factor'].values
    zoom_changes = np.abs(np.diff(zoom_factors))
    
    metrics = {
        'motion_analysis': {
            'max_speed_pixels_per_sec': float(np.max(speed)),
            'mean_speed_pixels_per_sec': float(np.mean(speed)),
            'max_acceleration': float(np.max(acceleration_magnitude)),
            'speed_consistency': float(1.0 - (np.std(speed) / np.mean(speed)) if np.mean(speed) > 0 else 1.0)
        },
        'zoom_analysis': {
            'average_zoom': float(np.mean(zoom_factors)),
            'zoom_range': f"{np.min(zoom_factors):.2f}x - {np.max(zoom_factors):.2f}x",
            'max_zoom_change_per_frame': float(np.max(zoom_changes)),
            'zoom_stability': float(1.0 - (np.std(zoom_changes) / np.mean(zoom_changes)) if np.mean(zoom_changes) > 0 else 1.0)
        },
        'frame_stats': {
            'total_frames': len(smoothed_df),
            'duration_seconds': len(smoothed_df) / fps,
            'fps': fps
        }
    }
    
    return metrics


def generate_smooth_ffmpeg_filter(smoothed_df: pd.DataFrame) -> str:
    """Build a valid sendcmd text (one option per line)."""
    if smoothed_df.empty:
        raise ValueError("No smoothed coordinates provided")

    rows: list[str] = []
    for _, r in smoothed_df.iterrows():
        t = r["t_ms"] / 1000.0
        w = int(r["crop_w"]) & ~1
        h = int(r["crop_h"]) & ~1
        x = int(r["crop_x"])
        y = int(r["crop_y"])
        rows.extend([
            f"{t:.3f} crop w {w}",
            f"{t:.3f} crop h {h}",
            f"{t:.3f} crop x {x}",
            f"{t:.3f} crop y {y}",
        ])
    return "\n".join(rows) 