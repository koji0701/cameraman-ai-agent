import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, Dict
import av


def interpolate_and_smooth_coordinates(
    normalized_crops: pd.DataFrame,
    video_path: str = None,
    target_fps: float = None,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic'
) -> pd.DataFrame:
    """
    Complete pipeline: Interpolate crop coordinates to per-frame and apply Kalman smoothing.
    
    Args:
        normalized_crops: DataFrame with columns ['t_ms', 'crop_x', 'crop_y', 'crop_w', 'crop_h']
        video_path: Path to video file (for FPS detection) 
        target_fps: Target frame rate (if video_path not provided)
        smoothing_strength: 'minimal', 'balanced', 'maximum', or 'cinematic'
        interpolation_method: 'cubic', 'linear', or 'quadratic'
        
    Returns:
        DataFrame with smoothed per-frame coordinates
    """
    
    if normalized_crops.empty:
        raise ValueError("No normalized crop coordinates provided")
    
    # Detect video FPS if not provided
    if target_fps is None:
        if video_path is None:
            target_fps = 30.0  # Default fallback
            print("Warning: No video path or FPS provided, using 30 FPS default")
        else:
            target_fps = detect_video_fps(video_path)
            print(f"Detected video FPS: {target_fps}")
    
    # Step 1: Interpolate coordinates to per-frame timeline
    print(f"Interpolating {len(normalized_crops)} keyframes to {target_fps} FPS...")
    interpolated_df = interpolate_coordinates_to_frames(
        normalized_crops, target_fps, interpolation_method
    )
    
    # Step 2: Apply Kalman filtering for smooth motion
    print(f"Applying Kalman smoothing ({smoothing_strength} mode)...")
    smoothed_df = apply_kalman_smoothing(
        interpolated_df, target_fps, smoothing_strength
    )
    
    # Step 3: Validate and constrain final coordinates
    smoothed_df = validate_and_constrain_coordinates(smoothed_df)
    
    print(f"Generated {len(smoothed_df)} smooth frames")
    return smoothed_df


def detect_video_fps(video_path: str) -> float:
    """Detect video frame rate using PyAV"""
    try:
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate)
            return fps
    except Exception as e:
        print(f"Warning: Could not detect FPS from {video_path}: {e}")
        return 30.0  # Default fallback


def interpolate_coordinates_to_frames(
    normalized_crops: pd.DataFrame, 
    fps: float,
    method: str = 'cubic'
) -> pd.DataFrame:
    """
    Interpolate crop coordinates from keyframes to per-frame timeline.
    """
    
    # Extract timestamps and coordinates
    timestamps_ms = normalized_crops['t_ms'].values
    coords = normalized_crops[['crop_x', 'crop_y', 'crop_w', 'crop_h']].values
    
    # Create per-frame timeline
    start_time = timestamps_ms.min()
    end_time = timestamps_ms.max()
    frame_interval_ms = 1000.0 / fps
    
    frame_timestamps = np.arange(start_time, end_time + frame_interval_ms, frame_interval_ms)
    
    # Interpolate each coordinate dimension
    interpolated_coords = np.zeros((len(frame_timestamps), 4))
    coord_names = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    
    for i, coord_name in enumerate(coord_names):
        if method == 'cubic':
            # Use cubic spline for smooth C² continuity
            spline = CubicSpline(
                timestamps_ms, coords[:, i], 
                bc_type='natural',  # Natural boundary conditions
                extrapolate=False
            )
            interpolated_coords[:, i] = spline(frame_timestamps)
            
        elif method == 'quadratic':
            # Quadratic spline for less aggressive smoothing
            from scipy.interpolate import interp1d
            interp_func = interp1d(timestamps_ms, coords[:, i], kind='quadratic', 
                                 fill_value='extrapolate')
            interpolated_coords[:, i] = interp_func(frame_timestamps)
            
        elif method == 'linear':
            # Linear interpolation for minimal processing
            interpolated_coords[:, i] = np.interp(frame_timestamps, timestamps_ms, coords[:, i])
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        't_ms': frame_timestamps,
        'frame_number': np.arange(len(frame_timestamps)),
        'crop_x': interpolated_coords[:, 0],
        'crop_y': interpolated_coords[:, 1], 
        'crop_w': interpolated_coords[:, 2],
        'crop_h': interpolated_coords[:, 3]
    })
    
    return result_df


def apply_kalman_smoothing(
    interpolated_df: pd.DataFrame,
    fps: float,
    smoothing_strength: str = 'balanced'
) -> pd.DataFrame:
    """
    Apply Kalman filtering to smooth coordinate trajectories.
    """
    
    # Configure Kalman filter parameters based on smoothing strength
    kalman_configs = {
        'minimal': {
            'process_noise': 0.1,
            'measurement_noise': 1.0,
            'description': 'Light smoothing, preserves original motion'
        },
        'balanced': {
            'process_noise': 0.05,
            'measurement_noise': 2.0,
            'description': 'Balanced smoothing for professional results'
        },
        'maximum': {
            'process_noise': 0.01,
            'measurement_noise': 5.0,
            'description': 'Heavy smoothing for very stable output'
        },
        'cinematic': {
            'process_noise': 0.02,
            'measurement_noise': 3.0,
            'description': 'Cinematic smoothing with natural motion'
        }
    }
    
    config = kalman_configs.get(smoothing_strength, kalman_configs['balanced'])
    print(f"Using {smoothing_strength} smoothing: {config['description']}")
    
    # Apply Kalman filtering to each coordinate dimension
    smoothed_coords = np.zeros_like(interpolated_df[['crop_x', 'crop_y', 'crop_w', 'crop_h']].values)
    coord_names = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    
    dt = 1.0 / fps  # Time step in seconds
    
    for i, coord_name in enumerate(coord_names):
        measurements = interpolated_df[coord_name].values
        
        # Validate measurements
        if not np.all(np.isfinite(measurements)):
            print(f"Warning: Non-finite values in {coord_name}, cleaning before filtering...")
            measurements = pd.Series(measurements).interpolate().fillna(method='ffill').fillna(method='bfill').values
        
        # Create and configure Kalman filter
        kf = create_position_velocity_kalman_filter(
            dt=dt,
            process_noise=config['process_noise'],
            measurement_noise=config['measurement_noise']
        )
        
        # Initialize filter with first measurement
        kf.x[0] = measurements[0]  # Initial position
        kf.x[1] = 0.0  # Initial velocity
        
        # Apply filter frame by frame
        for frame_idx in range(len(measurements)):
            # Predict step
            kf.predict()
            
            # Update with measurement
            kf.update(measurements[frame_idx])
            
            # Store smoothed position with bounds checking
            smoothed_position = float(kf.x[0])  # Extract scalar from array
            
            # Ensure the smoothed position is reasonable
            if not np.isfinite(smoothed_position):
                # Fall back to measurement if Kalman produces invalid result
                smoothed_position = measurements[frame_idx]
                print(f"Warning: Kalman produced non-finite value at frame {frame_idx}, using measurement")
            
            smoothed_coords[frame_idx, i] = smoothed_position
    
    # Create result DataFrame
    smoothed_df = interpolated_df.copy()
    smoothed_df[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = smoothed_coords
    
    return smoothed_df


def create_position_velocity_kalman_filter(
    dt: float,
    process_noise: float,
    measurement_noise: float
) -> KalmanFilter:
    """
    Create a Kalman filter for position-velocity tracking.
    
    State: [position, velocity]
    Measurement: position only
    """
    
    # Create filter: 2D state (position, velocity), 1D measurement (position)
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1.0, dt],   # position = position + velocity * dt
        [0.0, 1.0]   # velocity = velocity (constant)
    ])
    
    # Measurement function (observe position only)
    kf.H = np.array([[1.0, 0.0]])
    
    # Process noise covariance (how much we expect system to change)
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise)
    
    # Measurement noise covariance (how much we trust measurements)
    kf.R = np.array([[measurement_noise]])
    
    # Initial state covariance (high uncertainty initially)
    kf.P *= 100.0
    
    return kf


def validate_and_constrain_coordinates(
    smoothed_df: pd.DataFrame,
    frame_width: int = 1920,
    frame_height: int = 1080,
    min_crop_size: int = 100
) -> pd.DataFrame:
    """
    Validate and constrain smoothed coordinates to ensure valid video crops.
    """
    
    result_df = smoothed_df.copy()
    
    # Handle NaN and infinity values by replacing with previous valid values
    coord_cols = ['crop_x', 'crop_y', 'crop_w', 'crop_h']
    
    for col in coord_cols:
        # Replace NaN and infinity values
        mask = ~np.isfinite(result_df[col])
        if mask.any():
            print(f"Warning: Found {mask.sum()} non-finite values in {col}, interpolating...")
            result_df[col] = result_df[col].interpolate(method='linear')
            
            # If there are still NaN values at the beginning or end, forward/back fill
            result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN (shouldn't happen), use default values
            if col in ['crop_x', 'crop_y']:
                result_df[col] = result_df[col].fillna(0)
            else:  # crop_w, crop_h
                result_df[col] = result_df[col].fillna(min_crop_size)
    
    # Ensure integer coordinates after cleaning
    result_df[coord_cols] = result_df[coord_cols].round().astype(int)
    
    # Constrain coordinates to frame boundaries
    result_df['crop_x'] = np.clip(result_df['crop_x'], 0, frame_width)
    result_df['crop_y'] = np.clip(result_df['crop_y'], 0, frame_height)
    
    # Ensure minimum crop sizes
    result_df['crop_w'] = np.maximum(result_df['crop_w'], min_crop_size)
    result_df['crop_h'] = np.maximum(result_df['crop_h'], min_crop_size)
    
    # Ensure crop doesn't exceed frame boundaries
    result_df['crop_w'] = np.minimum(result_df['crop_w'], frame_width - result_df['crop_x'])
    result_df['crop_h'] = np.minimum(result_df['crop_h'], frame_height - result_df['crop_y'])
    
    # Ensure even dimensions for video encoders
    result_df['crop_w'] = result_df['crop_w'] - (result_df['crop_w'] % 2)
    result_df['crop_h'] = result_df['crop_h'] - (result_df['crop_h'] % 2)
    
    # Calculate zoom factors for analysis
    result_df['zoom_factor'] = np.minimum(
        frame_width / result_df['crop_w'],
        frame_height / result_df['crop_h']
    )
    
    return result_df


def analyze_motion_smoothness(smoothed_df: pd.DataFrame, fps: float) -> Dict:
    """
    Analyze the smoothness and quality of the motion after Kalman filtering.
    """
    
    if len(smoothed_df) < 3:
        return {"error": "Insufficient data for motion analysis"}
    
    # Calculate velocities (first derivative)
    dt = 1.0 / fps
    coords = smoothed_df[['crop_x', 'crop_y', 'crop_w', 'crop_h']].values
    velocities = np.diff(coords, axis=0) / dt
    
    # Calculate accelerations (second derivative)
    accelerations = np.diff(velocities, axis=0) / dt
    
    # Calculate motion metrics
    speed = np.sqrt(np.sum(velocities[:, :2]**2, axis=1))  # Only x,y position
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
    """
    Generate FFmpeg sendcmd filter from smoothed frame-by-frame coordinates.
    """
    
    if smoothed_df.empty:
        raise ValueError("No smoothed coordinates provided")
    
    crop_commands = []
    
    for _, row in smoothed_df.iterrows():
        timestamp = row['t_ms'] / 1000.0  # Convert to seconds
        x, y, w, h = int(row['crop_x']), int(row['crop_y']), int(row['crop_w']), int(row['crop_h'])
        
        # FFmpeg sendcmd format: timestamp [enter] cropfilter w h x y
        crop_commands.append(f"{timestamp:.3f} [enter] crop w {w} h {h} x {x} y {y};")
    
    return '\n'.join(crop_commands)


if __name__ == "__main__":
    # Test with normalized crop data
    test_normalized = pd.DataFrame({
        't_ms': [0, 3000, 6000, 9000, 12000],
        'crop_x': [146, 146, 143, 143, 121],
        'crop_y': [267, 267, 277, 277, 335],
        'crop_w': [737, 737, 792, 792, 586],
        'crop_h': [414, 414, 445, 445, 330]
    })
    
    print("Testing Kalman smoothing pipeline...")
    print(f"Input: {len(test_normalized)} keyframes")
    
    # Test interpolation and smoothing
    smoothed = interpolate_and_smooth_coordinates(
        test_normalized,
        target_fps=30.0,
        smoothing_strength='balanced'
    )
    
    print(f"Output: {len(smoothed)} smooth frames")
    
    # Analyze motion quality
    motion_metrics = analyze_motion_smoothness(smoothed, 30.0)
    print("\nMotion Analysis:")
    for category, metrics in motion_metrics.items():
        print(f"  {category}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")
    
    # Generate FFmpeg filter (first 10 frames)
    ffmpeg_filter = generate_smooth_ffmpeg_filter(smoothed.head(10))
    print(f"\nSample FFmpeg filter (first 10 frames):")
    print(ffmpeg_filter) 