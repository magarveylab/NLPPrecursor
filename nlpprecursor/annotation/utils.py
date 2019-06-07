
import torch


def sequence_mask(words: torch.Tensor, pad_idx: int) -> torch.ByteTensor:
    """
    Compute sequence mask.
    Parameters
    ----------
    lens : torch.Tensor
        Tensor of sequence lengths ``[batch_size]``.
    max_len : int, optional (default: None)
        The maximum length (optional).
    Returns
    -------
    torch.ByteTensor
        Returns a tensor of 1's and 0's of size ``[batch_size x max_len]``.
    """
    mask = words != pad_idx
    return mask.byte()