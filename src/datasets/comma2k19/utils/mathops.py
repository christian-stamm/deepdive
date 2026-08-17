import torch
from sympy import series


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) Hamilton quaternion (w,x,y,z)
    returns (...,3,3)
    """
    q = q / q.norm(dim=-1, keepdim=True)

    w, x, y, z = q.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    return R.reshape(*q.shape[:-1], 3, 3)


def ecef_to_ned(ecef_vel: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Convert ECEF velocity to NED velocity using the provided quaternion orientation.
    """

    # Convert quaternion to rotation matrix
    rotation_matrix = quat_to_rotmat(quat)

    # Transform ECEF velocity to NED velocity
    ned_velocity = rotation_matrix.transpose(-2, -1) @ ecef_vel.unsqueeze(-1)

    return ned_velocity.squeeze(-1)


def ecef_to_traj(ecef_pose: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Convert ECEF position to NED position using the provided quaternion orientation.
    """

    delta_epos = ecef_pose[1:] - ecef_pose[:-1]
    delta_body = ecef_to_ned(delta_epos, quat[0])
    delta_body = torch.cat(
        [
            torch.zeros(1, 3, device=delta_body.device, dtype=delta_body.dtype),
            delta_body,
        ],
        dim=0,
    )
    # delta_body = torch.cumsum(delta_body, dim=0)
    return delta_body


def consolidate_series(
    times: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(times)
    times = times[order]
    values = values[order]

    keep = torch.ones(times.size(0), dtype=torch.bool, device=times.device)
    keep[:-1] = times[:-1] != times[1:]

    return times[keep], values[keep]


def interpolate_series(
    target_times: torch.Tensor,  # (M,)
    source_times: torch.Tensor,  # (N,)
    source_values: torch.Tensor,  # (N, ...)
) -> torch.Tensor:
    """
    Interpolate source_values at target_times.

    Out-of-range target timestamps are clamped to the first/last
    source value.
    """
    if target_times.ndim != 1:
        raise ValueError("target_times must have shape (M,)")

    if source_times.ndim != 1:
        raise ValueError("source_times must have shape (N,)")

    if source_values.shape[0] != source_times.numel():
        raise ValueError("source_values.shape[0] must equal len(source_times)")

    if source_times.numel() < 2:
        raise ValueError("At least two source samples are required")

    order = torch.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]

    if not torch.all(source_times[:-1] < source_times[1:]):
        raise ValueError(
            "source_times must be strictly increasing; "
            "deduplicate timestamps before interpolation"
        )

    # First source timestamp >= each target timestamp.
    right_idx = torch.searchsorted(
        source_times,
        target_times,
        side="left",
    )

    # Valid interpolation pairs are (0, 1) through (N-2, N-1).
    right_idx = right_idx.clamp(1, source_times.numel() - 1)
    left_idx = right_idx - 1

    t0 = source_times[left_idx]  # (M,)
    t1 = source_times[right_idx]  # (M,)
    v0 = source_values[left_idx]  # (M, ...)
    v1 = source_values[right_idx]  # (M, ...)

    alpha = (target_times - t0) / (t1 - t0)
    alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)  # (M, 1)

    return torch.lerp(v0, v1, alpha)
