import pandas as pd
import numpy as np
from typing import Tuple


def normalize_bounding_boxes_to_video_resolution(
    df: pd.DataFrame, 
    original_width: int,
    original_height: int,
    target_width: int = 1920,
    target_height: int = 1080,
    padding_factor: float = 1.1,
    target_aspect_ratio: float = None
) -> pd.DataFrame:
    """Normalize bounding boxes from relative coordinates to video resolution with intelligent cropping."""
    
    if target_aspect_ratio is None:
        target_aspect_ratio = target_width / target_height
    
    normalized_crops = []
    
    for _, row in df.iterrows():
        x1, y1 = row['x'] * original_width, row['y'] * original_height
        x2, y2 = (row['x'] + row['width']) * original_width, (row['y'] + row['height']) * original_height
        
        bbox_width = (x2 - x1) * padding_factor
        bbox_height = (y2 - y1) * padding_factor
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        crop_width, crop_height = calculate_optimal_crop_size(
            bbox_width, bbox_height, target_aspect_ratio, original_width, original_height
        )
        
        crop_x = bbox_center_x - crop_width / 2
        crop_y = bbox_center_y - crop_height / 2
        
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


def calculate_optimal_crop_size(
    bbox_width: float, 
    bbox_height: float, 
    target_aspect_ratio: float,
    max_width: int,
    max_height: int
) -> Tuple[float, float]:
    """Calculate optimal crop size that contains the bounding box and maintains aspect ratio."""
    
    width_for_height = bbox_height * target_aspect_ratio
    height_for_width = bbox_width / target_aspect_ratio
    
    if width_for_height >= bbox_width:
        crop_width = width_for_height
        crop_height = bbox_height
    else:
        crop_width = bbox_width
        crop_height = height_for_width
    
    crop_width = min(crop_width, max_width)
    crop_height = min(crop_height, max_height)
    
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
    """Adjust crop position to ensure it stays within frame boundaries."""
    
    if crop_x < 0:
        crop_x = 0
    elif crop_x + crop_width > frame_width:
        crop_x = frame_width - crop_width
        
    if crop_y < 0:
        crop_y = 0
    elif crop_y + crop_height > frame_height:
        crop_y = frame_height - crop_height
        
    return crop_x, crop_y


def generate_ffmpeg_crop_filter(normalized_df: pd.DataFrame, fps: float = 30.0) -> str:
    """Generate FFmpeg crop filter string from normalized coordinates."""
    
    if normalized_df.empty:
        raise ValueError("No normalized coordinates provided")
    
    crop_commands = []
    
    for _, row in normalized_df.iterrows():
        timestamp = row['t_ms'] / 1000.0
        x, y, w, h = row['crop_x'], row['crop_y'], row['crop_w'], row['crop_h']
        
        w = int(w) - (int(w) % 2)
        h = int(h) - (int(h) % 2)
        x = int(x)
        y = int(y)
        
        crop_commands.append(f"{timestamp:.3f} crop=w={w}:h={h}:x={x}:y={y}")
    
    return '\n'.join(crop_commands)


def analyze_crop_quality(original_df: pd.DataFrame, normalized_df: pd.DataFrame) -> dict:
    """Analyze the quality of the normalization process."""
    
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