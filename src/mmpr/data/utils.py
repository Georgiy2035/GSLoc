import torch

def getBEV_torch(all_points, min_val=-30, max_val=30, resolution=0.2, device='cuda'):
    """
    all_points: torch.Tensor of shape (N, 3) on the specified device
    Returns: BEV image as torch.Tensor (H, W) on the same device
    """
    x_min, y_min = min_val, min_val
    x_max, y_max = max_val, max_val

    mask = (all_points[:,0].abs() < max_val) & (all_points[:,1].abs() < max_val) & (all_points[:,2].abs() < max_val)
    all_points = all_points[mask]

    x_min_ind = torch.floor(torch.tensor(x_min/resolution)).int()
    x_max_ind = torch.floor(torch.tensor(x_max/resolution)).int()
    y_min_ind = torch.floor(torch.tensor(y_min/resolution)).int()
    y_max_ind = torch.floor(torch.tensor(y_max/resolution)).int()
    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    mat_global_image = torch.zeros((y_num, x_num), dtype=torch.uint8, device=device)

    x_ind = x_max_ind - torch.floor(all_points[:, 1] / resolution).int()
    y_ind = y_max_ind - torch.floor(all_points[:, 0] / resolution).int()

    valid = (x_ind >= 0) & (x_ind < x_num) & (y_ind >= 0) & (y_ind < y_num)
    x_ind = x_ind[valid]
    y_ind = y_ind[valid]

    flat_indices = y_ind * x_num + x_ind
    bincount = torch.bincount(flat_indices, minlength=y_num * x_num)
    bincount = bincount.clamp(max=10).reshape(y_num, x_num)

    mat_global_image = bincount * 10
    mat_global_image = torch.clamp(mat_global_image, max=255)
    if mat_global_image.max() > 0:
        mat_global_image = mat_global_image.float() / mat_global_image.max() * 255
    else:
        mat_global_image = mat_global_image.float()
    mat_global_image = mat_global_image.to(torch.uint8)

    return mat_global_image