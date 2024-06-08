# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch

from vedacore.misc import registry
from .base_segment_coder import BaseSegmentCoder


@registry.register_module('segment_coder')
class DeltaSegmentCoder(BaseSegmentCoder):
    """Delta Segment coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes segment (start, end) into delta (d_center, d_interval)
    and decodes delta (d_center, d_interval) back to original segment
    (start, end).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self, target_means=(0., 0.), target_stds=(1., 1.)):
        super(BaseSegmentCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, segments, gt_segments):
        """Get segment regression transformation deltas that can be used to
        transform the ``segments`` into the ``gt_segments``.

        Args:
            segments (torch.Tensor): Source segments, e.g., object proposals.
            gt_segments (torch.Tensor): Target of the transformation, e.g.,
                ground-truth segments.

        Returns:
            torch.Tensor: segment transformation deltas
        """

        assert segments.size(0) == gt_segments.size(0)
        assert segments.size(-1) == gt_segments.size(-1) == 2
        encoded_segments = segment2delta(segments, gt_segments, self.means,
                                         self.stds)
        return encoded_segments

    def decode(self, segments, pred_segments, max_t=None):
        """Apply transformation `pred_segments` to `segments`.

        Args:
            segments (torch.Tensor): Basic segments.
            pred_segments (torch.Tensor): Encoded segments with shape
            max_t (int, optional): Maximum time of segments.
                Defaults to None.

        Returns:
            torch.Tensor: Decoded segments.
        """

        assert pred_segments.size(0) == segments.size(0)
        decoded_segments = delta2segment(segments, pred_segments, self.means,
                                         self.stds, max_t)

        return decoded_segments


def segment2delta(proposals, gt, means=(0., 0.), stds=(1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of center, interval of proposals w.r.t ground
    truth segments to get regression target.
    This is the inverse function of :func:`delta2segment`.

    Args:
        proposals (Tensor): Segments to be transformed, shape (N, ..., 2)
        gt (Tensor): Gt segments to be used as base, shape (N, ..., 2)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 2), where columns represent d_center,
            d_interval
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()

    p_center = (proposals[..., 0] + proposals[..., 1]) * 0.5
    p_interval = proposals[..., 1] - proposals[..., 0]

    g_center = (gt[..., 0] + gt[..., 1]) * 0.5
    g_interval = gt[..., 1] - gt[..., 0]

    d_center = (g_center - p_center) / p_interval
    d_interval = torch.log(g_interval / p_interval)
    deltas = torch.stack([d_center, d_interval], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2segment(rois, deltas, means=(0., 0.), stds=(1., 1.), max_t=None):
    """Apply deltas to shift/scale base segments.

    Typically the rois are anchor or proposed segments and the deltas are
    network outputs used to shift/scale those segments.
    This is the inverse function of :func:`segment2delta`.

    Args:
        rois (Tensor): Segments to be transformed. Has shape (N, 2)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 2 * num_classes). Note N = num_anchors * T when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_t (int): Maximum time for segments. specifies T

    Returns:
        Tensor: Segments with shape (N, 2), where columns represent
            start, end.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  1.],
        >>>                      [ 0.,  1.],
        >>>                      [ 0.,  1.],
        >>>                      [ 5., 5.]])
        >>> deltas = torch.Tensor([[  0.,   0.],
        >>>                        [  1.,   1.],
        >>>                        [  0.,   2.],
        >>>                        [ 0.7, -0.5]])
        >>> delta2segment(rois, deltas, max_t=32)
        tensor([[0.0000, 1.0000],
                [0.1409, 2.8591],
                [0.0000, 4.1945],
                [5.0000, 5.0000]])
    """
    # print("rois.shape:",rois.shape) #torch.Size([480, 2])
    # print("max_t:",max_t) #192
    # assert 1==2
    # print("deltas.shape:",deltas.shape) #torch.Size([480, 2])
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 2)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 2)
    denorm_deltas = deltas * stds + means
    # print("denorm_deltas.shape:",denorm_deltas.shape) #torch.Size([480, 2])
    d_center = denorm_deltas[:, 0::2]
    # print("d_center.shape:",d_center.shape) #torch.Size([480, 1]) 
    d_interval = denorm_deltas[:, 1::2]
    # print("d_interval.shape:",d_interval.shape) #torch.Size([480, 1]) 
    # Compute center of each roi
    p_center = ((rois[:, 0] + rois[:, 1]) *
                0.5).unsqueeze(1).expand_as(d_center)
    # print("p_center.shape:",p_center.shape) #torch.Size([480, 1]) 
    # Compute interval of each roi
    p_interval = (rois[:, 1] - rois[:, 0]).unsqueeze(1).expand_as(d_interval)
    # print("p_interval.shape:",p_interval.shape) #torch.Size([480, 1]) 
    # Use exp(network energy) to enlarge/shrink each roi
    g_interval = p_interval * d_interval.exp()
    # print("g_interval.shape:",g_interval.shape) #torch.Size([480, 1]) 
    # Use network energy to shift the center of each roi
    g_center = p_center + p_interval * d_center
    # print("g_center.shape:",g_center.shape) #torch.Size([480, 1]) 
    # Convert center-xy/width/height to top-left, bottom-right
    start = g_center - g_interval * 0.5
    end = g_center + g_interval * 0.5
    # print("start.shape:",start.shape) #torch.Size([480, 1]) 
    # print("end.shape:",end.shape) #torch.Size([480, 1]) 
    if max_t is not None:
        # print("max_t is not None") #执行
        start = start.clamp(min=0, max=max_t)
        end = end.clamp(min=0, max=max_t)
    segments = torch.stack([start, end], dim=-1).view_as(deltas)
    # print("segments.shape:",segments.shape) #torch.Size([480, 2])
    # assert 1==2
    return segments
