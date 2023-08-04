import numpy as np
import torch

def maybe_cast_to_list(x):
    if isinstance(x, np.ndarray):
        return [item.tolist() for item in x]
    return x

def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m

@torch.no_grad()
def create_attention_mask(max_length):
    """Create `attention_mask`.
    Args:
        input_ids: A 1D tensor that holds the indices of tokens.
    """
    # seq_length = len(input_ids)
    # `attention_mask` has the shape of [1, seq_length, seq_length]
    attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
    attention_mask = attention_mask < 0.5
    return attention_mask
