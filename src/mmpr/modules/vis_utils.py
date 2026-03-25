from typing import Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def quaternion_angle(q1: Union[np.ndarray, list, tuple], 
                     q2: Union[np.ndarray, list, tuple], 
                     degrees: bool = True,
                     normalize: bool = True) -> float:
    
    """2 quaternion difference angle in degrees.
    
    Args:
        q1, q2: array-like quternions (x, y, z, w)
        degrees: bool value, sets the output units (degree if True, else radians)
        normalize: bool value, normalize quaternions if True
        
    Returns:
        angle : float quaternion rotation angle
    """
    # Convertion in numpy array
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    
    # Dimension check
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError(f"Quaternion dimension is need to be 4. Got: {q1.shape}, {q2.shape}")
    
    # Automatic order checking (w, x, y, z) or (x, y, z, w)
    # If scalar part (w) absolute value more that in other parts
    if abs(q1[0]) > max(abs(q1[1]), abs(q1[2]), abs(q1[3])):
        # (w, x, y, z) order - remain
        pass
    elif abs(q1[3]) > max(abs(q1[0]), abs(q1[1]), abs(q1[2])):
        # (x, y, z, w) order - change to (w, x, y, z)
        q1 = np.roll(q1, -1)  # [x, y, z, w] -> [w, x, y, z]
        q2 = np.roll(q2, -1)
    
    if normalize:
        # quaternions normalization
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
    
    # Dot product calculation
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    # Angle calculation (using absolute value because q and -q are the same rotation)
    angle = 2.0 * np.arccos(abs(dot))
    
    # Degree conversion
    if degrees:
        angle = np.degrees(angle)
    
    return float(angle)

def add_frame(image, frame_color=(255, 0, 0), frame_width=50):
    """Add a solid color frame around an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        frame_width: Width of the frame in pixels
        frame_color: RGB tuple for frame color
    """
    # Open the image
    img = image
    
    # Create a new image with frame dimensions
    new_width = img.shape[1] + (2 * frame_width)
    new_height = img.shape[0] + (2 * frame_width)
    
    # Create new image with frame color
    framed_img = Image.new('RGB', (new_width, new_height), color=tuple(frame_color))
    
    # Paste original image in the center
    framed_img.paste(Image.fromarray(img), box=(frame_width, frame_width)) #new_width + frame_width, new_height + frame_width))
    
    arr = np.asarray(framed_img, dtype=np.uint8)
    return arr

def add_feature_map(main_img: np.ndarray, 
                      feature_map: np.ndarray, 
                      thumb_scale: float = 0.35,
                      ) -> np.ndarray:
    
    """Insert small image in the corner of base image using PIL.
    
    Args:
        main_img: np.ndarray base image with (H, W, C) or (H, W) dimensions 
        feature_map: np.ndarray feature_map image with (H, W, C) or (H, W) dimensions
        thumb_scale: float scale relative to short side of base image
        
    Returns:
        result_img_np: np.ndarray result base image with feature map in the corner
    """

    # Convert numpy array into PIL Image
    if main_img.ndim == 2:
        main_pil = Image.fromarray(main_img).convert('RGB')
    else:
        main_pil = Image.fromarray(main_img)
    
    if feature_map.ndim == 2:
        thumb_pil = Image.fromarray(feature_map).convert('RGB')
    else:
        thumb_pil = Image.fromarray(feature_map)
    
    # Base image sides
    main_width, main_height = main_pil.size
    
    # feature map sides
    thumb_width = int(main_width * thumb_scale)
    thumb_height = int(main_height * thumb_scale)
    
    # Изменяем размер миниатюры с сохранением пропорций
    thumb_pil.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    thumb_width, thumb_height = thumb_pil.size
    
    # Создаем копию основного изображения
    result_pil = main_pil.copy()
    
    # Вычисляем координаты для вставки (правый нижний угол с отступом)
    x_pos = main_width - thumb_width
    y_pos = main_height - thumb_height
    
    thumb_with_border = thumb_pil
    
    # Вставляем миниатюру в основное изображение
    if thumb_with_border.mode == 'RGBA':
        result_pil.paste(thumb_with_border, (x_pos, y_pos), thumb_with_border)
    else:
        result_pil.paste(thumb_with_border, (x_pos, y_pos))
    
    # Преобразуем обратно в numpy array
    result_img_np = np.array(result_pil, dtype=np.uint8)
    
    return result_img_np

def plot_alignment(
    db_map_pcd,
    minimap_pts,
    query_pts_init,
    query_pts_refined,
    info_text=None,
    bounds_margin=2.0,
):
    """Plot base map (cropped), mini-map, and query before/after refinement in XY."""
    # Compute focused bounds
    stack = [minimap_pts[:, :2]]
    if query_pts_init is not None:
        stack.append(query_pts_init[:, :2])
    if query_pts_refined is not None:
        stack.append(query_pts_refined[:, :2])
    all_xy = np.vstack(stack)
    x_min, x_max = (
        all_xy[:, 0].min() - bounds_margin,
        all_xy[:, 0].max() + bounds_margin,
    )
    y_min, y_max = (
        all_xy[:, 1].min() - bounds_margin,
        all_xy[:, 1].max() + bounds_margin,
    )

    # Filter base map to focused region
    mask = (
        (db_map_pcd[:, 0] >= x_min)
        & (db_map_pcd[:, 0] <= x_max)
        & (db_map_pcd[:, 1] >= y_min)
        & (db_map_pcd[:, 1] <= y_max)
    )
    db_crop = db_map_pcd[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.scatter(db_crop[:, 0], db_crop[:, 1], s=1, alpha=1, label="DB map (crop)")
    ax.scatter(
        minimap_pts[:, 0], minimap_pts[:, 1], s=1, alpha=0.8, c='k', label="Mini‑map (merged)"
    )
    if query_pts_init is not None:
        ax.scatter(
            query_pts_init[:, 0],
            query_pts_init[:, 1],
            s=1,
            alpha=0.5,
            c='r',
            label="Query (init)",
        )
    if query_pts_refined is not None:
        ax.scatter(
            query_pts_refined[:, 0],
            query_pts_refined[:, 1],
            s=1,
            alpha=0.3,
            c='b',
            label="Query (refined)",
        )

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Query Alignment on Mini‑map", pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    if info_text:
        ax.text(
            0.01,
            0.99,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )
    plt.tight_layout()
    plt.show()
