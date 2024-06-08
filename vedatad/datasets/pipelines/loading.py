# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import glob
import os.path as osp

import numpy as np

import vedacore.fileio as fileio
import vedacore.image as image
from vedacore.misc import registry
import cv2
# import mmcv
import torch
import torchvision.transforms.functional as F
from PIL import Image
import io
@registry.register_module('pipeline')
class LoadMetaInfo(object):

    def __call__(self, results):
        results['img_ids'] = list(range(results['video_info']['frames']))
        results['ori_tsize'] = results['video_info']['frames']
        results['tsize'] = results['video_info']['frames']
        results['fps'] = results['video_info']['fps']
        results['duration'] = results['video_info']['duration']

        return results
from typing import List, Optional, Tuple, Union, no_type_check
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)
def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size
def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    backend = 'cv2'
    if backend is None:
        backend = imread_backend
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

# import psutil
@registry.register_module('pipeline')
class LoadFrames(object):
    """Load video frames.

    Required keys are "video_prefix" and "video_info" (a dict that must contain
    the key "video_name"). Added or updated keys are "video_name", "imgs",
    "imgs_shape", "ori_shape" (same as `imgs_shape`).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`image.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`fileio.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,   
                 color_type='color',
                 frame_resize=(171, 128),
                 keep_ratio=False,
                 use_resize=True,
                 file_client_args=dict(backend='disk')):  
        self.to_float32 = to_float32
        self.color_type = color_type

        if isinstance(frame_resize, float):
            if frame_resize <= 0:
                raise ValueError(f'Invalid frame resize {frame_resize}, must be positive.')
        elif isinstance(frame_resize, tuple):
            max_long_edge = max(frame_resize)
            max_short_edge = min(frame_resize)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                frame_resize = (np.inf, max_long_edge)

        self.frame_resize = frame_resize
        self.keep_ratio = keep_ratio
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.use_resize=use_resize

    def __call__(self, results):
        """Call functions to load frames and get frames meta information.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded frames and meta information.
        """

        if self.file_client is None:
            self.file_client = fileio.FileClient(**self.file_client_args)

        if results['video_prefix'] is not None:
            video_name = osp.join(results['video_prefix'],
                                  results['video_info']['video_name'])
        else:
            video_name = results['video_info']['video_name']

        imgfiles = sorted(glob.glob(osp.join(video_name, '*')))
        assert len(imgfiles) == results['video_info']['frames'] 
        imgs = []
        for img_id in results['img_ids']:
            if isinstance(img_id, int)==False:
                img_id=int(img_id)
            filename = imgfiles[img_id]
            img_bytes = self.file_client.get(filename)
            img = image.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)

            img_h, img_w = img.shape[:2]

            if self.keep_ratio:
                new_w, new_h = rescale_size((img_w, img_h), self.frame_resize)
            else:
                new_w, new_h = self.frame_resize
            if self.use_resize:
                img = imresize(img, (new_w, new_h), interpolation="bilinear")

            imgs.append(img)
        imgs = np.array(imgs)


        results['video_name'] = video_name
        results['ori_video_name'] = results['video_info']['video_name']
        results['imgs'] = imgs
        results['tsize'] = imgs.shape[0]
        results['pad_tsize'] = imgs.shape[0]
        results['img_fields'] = ['imgs']

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@registry.register_module('pipeline')
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_segment (bool): Whether to parse and load the segment annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`fileio.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_segment=True,
                 with_label=True,
                 file_client_args=dict(backend='disk')):
        self.with_segment = with_segment
        self.with_label = with_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_segments(self, results):
        """Private function to load segment annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded segment annotations.
        """

        ann_info = results['ann_info']
        results['gt_segments'] = ann_info['segments'].copy()

        gt_segments_ignore = ann_info.get('segments_ignore', None)
        if gt_segments_ignore is not None:
            results['gt_segments_ignore'] = gt_segments_ignore.copy()
            results['segment_fields'].append('gt_segments_ignore')
        results['segment_fields'].append('gt_segments')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded segment, label annotations.
        """

        if self.with_segment:
            results = self._load_segments(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_segment={self.with_segment}, '
        repr_str += f'with_label={self.with_label})'
        return repr_str


@registry.register_module('pipeline')
class Time2Frame(object):
    """Switch time point to frame index."""

    def __call__(self, results):
        """Call function to switch time point to frame index.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Switch results.
        """

        for key in results.get('segment_fields', []):
            results[key] = results[key] * results['fps']

        return results

@registry.register_module('pipeline')
class PyAVInit(object):
    """Using pyav to initialize the video.
    PyAV: https://github.com/mikeboers/PyAV
    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".
    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = fileio.FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['video_info']['filename'])) 
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str

@registry.register_module('pipeline')
class PyAVDecode(object):
    """Using PyAV to decode the video.
    PyAV: https://github.com/mikeboers/PyAV
    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".
    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            the nearest key frames, which may be duplicated and inaccurate,
            and more suitable for large scene-based video datasets.
            Default: 'accurate'.
    """

    def __init__(self,fps,to_float32=False,frame_resize=(171, 128),keep_ratio=False, use_resize=True,multi_thread=False, mode='accurate',color_type='color',file_client_args=dict(backend='disk')):
        self.multi_thread = multi_thread
        self.mode = mode
        self.fps = fps
        self.to_float32 = to_float32
        self.color_type = color_type
        self.keep_ratio = keep_ratio
        assert mode in ['accurate', 'efficient']

        if isinstance(frame_resize, float):
            if frame_resize <= 0:
                raise ValueError(f'Invalid frame resize {frame_resize}, must be positive.')
        elif isinstance(frame_resize, tuple):
            max_long_edge = max(frame_resize)
            max_short_edge = min(frame_resize)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                frame_resize = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'frame resize must be float or tuple of int, but got {type(frame_resize)}')
        self.frame_resize = frame_resize
        self.use_resize=use_resize

        self.file_client_args = file_client_args.copy()
        self.file_client = None

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def __call__(self, results):
        """Perform the PyAV decoding.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = fileio.FileClient(**self.file_client_args)
        file_obj = io.BytesIO(self.file_client.get(results['video_info']['filename'])) 
        container = av.open(file_obj)
        imgs = list()
        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'

        if self.mode == 'accurate':
            # set max indice to make early stop
            max_inds = results['video_info']['frames'] 
            start = results['img_ids'][0]
            end = results['img_ids'][-1]
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                if i > end:
                    break
                if i < start:
                    i+=1
                    continue
                img = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR) 

                if self.to_float32:
                    img = img.astype(np.float32)
                img_h, img_w = img.shape[:2]

                if self.keep_ratio:
                    new_w, new_h = rescale_size((img_w, img_h), self.frame_resize) 
                else:
                    new_w, new_h = self.frame_resize
                if self.use_resize:
                    img = imresize(img, (new_w, new_h), interpolation="bilinear") 
                imgs.append(img)
                i += 1
            imgs = np.array(imgs)

        results['video_name'] = results['video_info']['video_name'] 
        results['ori_video_name'] = results['video_info']['video_name']
        results['imgs'] = imgs 
        results['tsize'] = imgs.shape[0]
        results['pad_tsize'] = imgs.shape[0]
        results['img_fields'] = ['imgs']

        del container

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str

