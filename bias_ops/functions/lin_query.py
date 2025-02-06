import torch
import biasops._C
from torch import dtype


@torch.inference_mode()
def offset_to_bincount(offset):
    """
    Convert an offset tensor to its bincount representation.
    """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset_to_batch(offset):
    """
    Generate a batch index tensor from an offset tensor.
    """
    bincount = offset_to_bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch_to_offset(batch):
    """
    Convert a batch tensor to its offset representation.
    """
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(matrix):
    """
    Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = matrix.shape
    assert n == m, "Input matrix must be square."
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def xyz_to_key(new_xyz):
    """
    Convert a 2D tensor with shape (N, 4) into a unique hash key using bit-shifting.
    """
    assert new_xyz.dim() == 2 and new_xyz.size(1) == 4, \
        "Input must be a 2D tensor with shape (N, 4)."

    # Define bit sizes for each dimension
    bits_x, bits_y, bits_z, bits_w = 18, 18, 18, 8

    # Define shift values for bit manipulation
    shift_x = bits_y + bits_z + bits_w
    shift_y = bits_z + bits_w
    shift_z = bits_w
    shift_w = 0

    # Define masks for extracting bits
    mask_x = (1 << bits_x) - 1
    mask_y = mask_x
    mask_z = mask_x
    mask_w = (1 << bits_w) - 1

    # Apply masks and shifts to generate keys
    x = new_xyz[:, 0] & mask_x
    y = new_xyz[:, 1] & mask_y
    z = new_xyz[:, 2] & mask_z
    w = new_xyz[:, 3] & mask_w

    return (x << shift_x) | (y << shift_y) | (z << shift_z) | w


def query(p, new_p, offset, new_offset, grid_size=0.1, kernel_size=3, index=None):
    """
    Query neighbors within a grid and generate a rulebook for kernel operations.

    Args:
        p (Tensor): Original point cloud coordinates.
        new_p (Tensor): New point cloud coordinates.
        offset (Tensor): Offset tensor for `p`.
        new_offset (Tensor): Offset tensor for `new_p`.
        grid_size (float): Grid cell size.
        kernel_size (int): Kernel size for neighbor search.
        index (Tensor, optional): Index mapping for `output_idx`.

    Returns:
        Tensor: Rulebook containing input, output, and potentially transformed indices.
    """
    # Normalize points into grid space
    p_grid = torch.div(
        p - torch.min(p, dim=0, keepdim=True)[0],
        grid_size,
        rounding_mode='floor'
    )
    new_p_grid = torch.div(
        new_p - torch.min(p, dim=0, keepdim=True)[0],
        grid_size,
        rounding_mode='floor'
    )

    # Append batch indices
    batch = offset_to_batch(offset).unsqueeze(-1)
    new_batch = offset_to_batch(new_offset).unsqueeze(-1)

    p_grid = torch.cat([p_grid, batch], dim=-1).long()
    new_p_grid = torch.cat([new_p_grid, new_batch], dim=-1).long()

    # Generate hash table
    key = xyz_to_key(p_grid)
    hash_table_keys, hash_table_values = biasops._C.generate_hash_table(key)

    # Compute neighbor counts and offsets
    counts = biasops._C.compute_counts(kernel_size, new_p_grid, hash_table_keys, hash_table_values)[0]
    offsets = torch.zeros(counts.shape[0] + 1, dtype=torch.int64, device=counts.device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    # Fill neighbor indices
    input_idx, output_idx = biasops._C.fill_neighbor_indices(
        kernel_size,
        new_p_grid,
        hash_table_keys,
        hash_table_values,
        counts,
        offsets
    )

    # Generate rulebook
    if index is None:
        rulebook = torch.stack([input_idx, output_idx, output_idx], dim=1)
    else:
        output_idx_p = index[output_idx]
        rulebook = torch.stack([input_idx, output_idx, output_idx_p], dim=1)

    return rulebook
