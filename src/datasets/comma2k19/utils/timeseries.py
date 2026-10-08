import torch

def consolidate_series(
    times: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(times)
    times = times[order]
    values = values[order]

    no_duplicate_mask = torch.ones(times.size(0), dtype=torch.bool, device=times.device)
    no_duplicate_mask[:-1] = times[:-1] != times[1:]

    return times[no_duplicate_mask], values[no_duplicate_mask]


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
