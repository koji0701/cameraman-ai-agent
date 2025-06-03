import pandas as pd
import numpy as np
from typing import Tuple, Union


def normalize_bounding_boxes_to_video_resolution(
    df: pd.DataFrame, 
    original_width: int,
    original_height: int,
    target_aspect_ratio: float = 16/9,  # Default to 16:9 but can be overridden
    padding_factor: float = 1.1
) -> pd.DataFrame:
    """
    Normalize bounding boxes to optimal crop coordinates while preserving original video resolution.
    
    Args:
        df: DataFrame with columns ['t_ms', 'x1', 'y1', 'x2', 'y2']
        original_width: Width of the source video frame
        original_height: Height of the source video frame  
        target_aspect_ratio: Desired aspect ratio for crops (default 16:9)
        padding_factor: Additional padding around bounding box (1.1 = 10% padding)
        
    Returns:
        DataFrame with normalized crop coordinates ['t_ms', 'crop_x', 'crop_y', 'crop_w', 'crop_h']
    """
    
    print(f"🎯 Normalizing coordinates for {original_width}x{original_height} video")
    print(f"   Target aspect ratio: {target_aspect_ratio:.2f}:1")
    print(f"   Padding factor: {padding_factor}")
    
    normalized_crops = []
    
    for _, row in df.iterrows():
        # Extract bounding box coordinates
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        
        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bounding box at t={row['t_ms']}ms: ({x1},{y1},{x2},{y2})")
            continue
            
        # Calculate bounding box dimensions with padding
        bbox_width = (x2 - x1) * padding_factor
        bbox_height = (y2 - y1) * padding_factor
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Calculate optimal crop dimensions that maintain aspect ratio
        crop_width, crop_height = calculate_optimal_crop_size(
            bbox_width, bbox_height, target_aspect_ratio, original_width, original_height
        )
        
        # Calculate crop position centered on bounding box
        crop_x = bbox_center_x - crop_width / 2
        crop_y = bbox_center_y - crop_height / 2
        
        # Adjust crop position to stay within frame boundaries
        crop_x, crop_y = constrain_crop_to_frame(
            crop_x, crop_y, crop_width, crop_height, original_width, original_height
        )
        
        normalized_crops.append({
            't_ms': row['t_ms'],
            'crop_x': int(crop_x),
            'crop_y': int(crop_y), 
            'crop_w': int(crop_width),
            'crop_h': int(crop_height),
            'zoom_factor': min(original_width / crop_width, original_height / crop_height)
        })
    
    return pd.DataFrame(normalized_crops)


def normalize_bounding_boxes_to_1080p(
    df: pd.DataFrame, 
    original_width: int = 1920, 
    original_height: int = 1080,
    target_width: int = 1920,
    target_height: int = 1080,
    padding_factor: float = 1.1
) -> pd.DataFrame:
    """
    DEPRECATED: Use normalize_bounding_boxes_to_video_resolution instead.
    
    This function is maintained for backward compatibility but should not be used
    for new implementations as it assumes fixed 1920x1080 resolution.
    """
    print("⚠️ normalize_bounding_boxes_to_1080p is deprecated.")
    print("⚠️ Use normalize_bounding_boxes_to_video_resolution for proper resolution handling.")
    
    target_aspect_ratio = target_width / target_height
    return normalize_bounding_boxes_to_video_resolution(
        df, original_width, original_height, target_aspect_ratio, padding_factor
    )


def calculate_optimal_crop_size(
    bbox_width: float, 
    bbox_height: float, 
    target_aspect_ratio: float,
    max_width: int,
    max_height: int
) -> Tuple[float, float]:
    """
    Calculate optimal crop size that contains the bounding box and maintains aspect ratio.
    
    Returns the smallest possible crop size with correct aspect ratio that fits the bbox.
    """
    
    # Calculate required dimensions to fit bbox with target aspect ratio
    width_for_height = bbox_height * target_aspect_ratio
    height_for_width = bbox_width / target_aspect_ratio
    
    if width_for_height >= bbox_width:
        # Height constraint is limiting - use bbox height and calculate width
        crop_width = width_for_height
        crop_height = bbox_height
    else:
        # Width constraint is limiting - use bbox width and calculate height  
        crop_width = bbox_width
        crop_height = height_for_width
    
    # Ensure crop doesn't exceed original frame size
    crop_width = min(crop_width, max_width)
    crop_height = min(crop_height, max_height)
    
    # Re-adjust to maintain exact aspect ratio after clamping
    if crop_width / crop_height > target_aspect_ratio:
        crop_width = crop_height * target_aspect_ratio
    else:
        crop_height = crop_width / target_aspect_ratio
        
    return crop_width, crop_height


def constrain_crop_to_frame(
    crop_x: float, 
    crop_y: float, 
    crop_width: float, 
    crop_height: float,
    frame_width: int, 
    frame_height: int
) -> Tuple[float, float]:
    """
    Adjust crop position to ensure it stays within frame boundaries.
    """
    
    # Constrain X position
    if crop_x < 0:
        crop_x = 0
    elif crop_x + crop_width > frame_width:
        crop_x = frame_width - crop_width
        
    # Constrain Y position  
    if crop_y < 0:
        crop_y = 0
    elif crop_y + crop_height > frame_height:
        crop_y = frame_height - crop_height
        
    return crop_x, crop_y


def generate_ffmpeg_crop_filter(normalized_df: pd.DataFrame, fps: float = 30.0) -> str:
    """
    Generate FFmpeg crop filter string from normalized coordinates.
    
    Args:
        normalized_df: DataFrame with crop coordinates
        fps: Video frame rate for timestamp conversion
        
    Returns:
        FFmpeg filter string for dynamic cropping
    """
    
    if normalized_df.empty:
        raise ValueError("No normalized coordinates provided")
    
    crop_commands = []
    
    for _, row in normalized_df.iterrows():
        timestamp = row['t_ms'] / 1000.0  # Convert to seconds
        x, y, w, h = row['crop_x'], row['crop_y'], row['crop_w'], row['crop_h']
        
        # Ensure even dimensions for video encoding
        w = int(w) - (int(w) % 2)
        h = int(h) - (int(h) % 2)
        x = int(x)
        y = int(y)
        
        crop_commands.append(f"{timestamp:.3f} crop=w={w}:h={h}:x={x}:y={y}")
    
    return '\n'.join(crop_commands)


def analyze_crop_quality(original_df: pd.DataFrame, normalized_df: pd.DataFrame) -> dict:
    """
    Analyze the quality of the normalization process.
    
    Returns metrics about zoom factors, coverage, etc.
    """
    
    if normalized_df.empty:
        return {"error": "No normalized data to analyze"}
    
    zoom_factors = normalized_df['zoom_factor'].values
    
    metrics = {
        'average_zoom': float(np.mean(zoom_factors)),
        'min_zoom': float(np.min(zoom_factors)), 
        'max_zoom': float(np.max(zoom_factors)),
        'zoom_std': float(np.std(zoom_factors)),
        'total_frames': len(normalized_df),
        'zoom_consistency': float(1.0 - (np.std(zoom_factors) / np.mean(zoom_factors))),
        'recommended_zoom_range': f"{np.min(zoom_factors):.2f}x - {np.max(zoom_factors):.2f}x"
    }
    
    return metrics


if __name__ == "__main__":
    # Test with your example data
    test_data = {
        't_ms': [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000],
        'x1': [180, 180, 180, 180, 180, 180, 180, 180, 180, 180],
        'y1': [350, 350, 350, 350, 350, 350, 350, 350, 350, 350], 
        'x2': [850, 850, 900, 900, 900, 700, 650, 650, 650, 650],
        'y2': [600, 600, 650, 650, 650, 650, 650, 650, 650, 650]
    }
    
    df = pd.DataFrame(test_data)
    print("Original bounding boxes:")
    print(df)
    
    normalized = normalize_bounding_boxes_to_video_resolution(df, 1920, 1080)
    print("\nNormalized crop coordinates:")
    print(normalized)
    
    quality_metrics = analyze_crop_quality(df, normalized)
    print("\nCrop quality analysis:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value}")
    
    print("\nFFmpeg crop filter:")
    print(generate_ffmpeg_crop_filter(normalized)) 