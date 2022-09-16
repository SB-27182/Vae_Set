import torch


def sum_except_batch(x, sum_up_to_this=1):
    """Sums all dimensions of the input tensor x,
        except for all dimensions up to and including the argument (default:1)."""

    return x.reshape(*x.shape[:sum_up_to_this], -1).sum(-1)