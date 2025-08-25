"""
Metrics computation utilities for GAOT.
"""
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from src.datasets.dataset import Metadata, SIMULATION_VARIABLES

EPSILON = 1e-10

def compute_batch_errors(gtr: torch.Tensor, prd: torch.Tensor, metadata: Metadata, simulation_name: str) -> torch.Tensor:
    """
    Compute the per-sample relative L1 errors per variable chunk for a batch.
    
    Args:
        gtr (torch.Tensor): Ground truth tensor with shape [batch_size, time, space, var]
        prd (torch.Tensor): Predicted tensor with shape [batch_size, time, space, var]
        metadata (Metadata): Dataset metadata including global_mean, global_std, and variable chunks
        simulation_name (str): Name of current simulation to identify active variables
    
    Returns:
        torch.Tensor: Relative errors per sample per variable chunk, shape [batch_size, num_chunks]
    """
    sim_vars = SIMULATION_VARIABLES[simulation_name]

    num_active_vars = len(sim_vars["outputs"])

    mean = torch.tensor(metadata.global_mean[:num_active_vars], device=gtr.device, dtype=gtr.dtype).reshape(1, 1, 1, -1)
    std = torch.tensor(metadata.global_std[:num_active_vars], device=gtr.device, dtype=gtr.dtype).reshape(1, 1, 1, -1)

    all_output_vars = sorted(list(set(var for sim in SIMULATION_VARIABLES.values() for var in sim['outputs'])))
    output_var_to_index = {var: i for i, var in enumerate(all_output_vars)}

    active_indices = [output_var_to_index[var] for var in sim_vars["outputs"]]

    gtr_active = gtr[..., active_indices]
    prd_active = prd[..., active_indices]

    gtr_norm = (gtr - mean) / std
    prd_norm = (prd - mean) / std

    original_chunks = metadata.chunked_variables
    chunked_vars = [original_chunks[i] for i in range(num_active_vars)]
    unique_chunks = sorted(set(chunked_vars))
    chunk_map = {old_chunk: new_chunk for new_chunk, old_chunk in enumerate(unique_chunks)}
    adjusted_chunks = [chunk_map[chunk] for chunk in chunked_vars]
    num_chunks = len(unique_chunks)

    chunks = torch.tensor(adjusted_chunks, device=gtr.device, dtype=torch.long)  # Shape: [var]

    # compute absolute errors and sum over the time and space dimensions
    abs_error = torch.abs(gtr_norm - prd_norm)  # Shape: [batch_size, time, space, var]
    error_sum = torch.sum(abs_error, dim=(1, 2))  # Shape: [batch_size, var]

    # sum errors per variable chunk
    chunks_expanded = chunks.unsqueeze(0).expand(error_sum.size(0), -1)  # Shape: [batch_size, var]
    error_per_chunk = torch.zeros(error_sum.size(0), num_chunks, device=gtr.device, dtype=error_sum.dtype)
    error_per_chunk.scatter_add_(1, chunks_expanded, error_sum)

    # compute sum of absolute values of the ground truth per chunk
    gtr_abs_sum = torch.sum(torch.abs(gtr_norm), dim=(1, 2))  # Shape: [batch_size, var]
    gtr_sum_per_chunk = torch.zeros(gtr_abs_sum.size(0), num_chunks, device=gtr.device, dtype=gtr_abs_sum.dtype)
    gtr_sum_per_chunk.scatter_add_(1, chunks_expanded, gtr_abs_sum)

    # compute relative errors per chunk
    relative_error_per_chunk = error_per_chunk / (gtr_sum_per_chunk + EPSILON) # Shape: [batch_size, num_chunks]

    return relative_error_per_chunk # Shape: [batch_size, num_chunks]

def compute_final_metric(all_relative_errors: torch.Tensor) -> float:
    """
    Compute the final metric from the accumulated relative errors.
    
    Args:
        all_relative_errors (torch.Tensor): Tensor of shape [num_samples, num_chunks]
        
    Returns:
        Metrics: An object containing the final relative L1 median error
    """
    # Compute the median over the sample axis for each chunk
    median_error_per_chunk = torch.median(all_relative_errors, dim=0)[0]  # Shape: [num_chunks]

    # Take the mean of the median errors across all chunks
    final_metric = torch.mean(median_error_per_chunk)
    
    return final_metric.item()

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pred_active = pred[..., mask]
    target_active = target[..., mask]
    return F.mse_loss(pred_active, target_active)
