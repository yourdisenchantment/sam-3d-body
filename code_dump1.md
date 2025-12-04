* Файл: sam_3d_body/build_models.py

```python

import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict

def load_sam_3d_body(
    checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""
):
    print("Loading SAM 3D Body model...")


    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):

        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)


    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()


    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_cfg

def _hf_download(repo_id):
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(
        local_dir, "assets", "mhr_model.pt"
    )

def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)

```

* Файл: sam_3d_body/data/__init__.py

```python

```

* Файл: sam_3d_body/data/transforms/bbox_utils.py

```python

import math
from typing import Tuple

import cv2
import numpy as np

def bbox_xyxy2xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh

def bbox_xywh2xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy

def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """

    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def bbox_xywh2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (x, y, h, w)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """


    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x, y, w, h = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x + w * 0.5, y + h * 0.5])
    scale = np.hstack([w, h]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def bbox_cs2xyxy(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x1,y1,x2,y2).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x1, y1, x2, y2) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, xy + wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox

def bbox_cs2xywh(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x,y,w,h).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x, y, w, h) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox

def flip_bbox(
    bbox: np.ndarray,
    image_size: Tuple[int, int],
    bbox_format: str = "xywh",
    direction: str = "horizontal",
) -> np.ndarray:
    """Flip the bbox in the given direction.

    Args:
        bbox (np.ndarray): The bounding boxes. The shape should be (..., 4)
            if ``bbox_format`` is ``'xyxy'`` or ``'xywh'``, and (..., 2) if
            ``bbox_format`` is ``'center'``
        image_size (tuple): The image shape in [w, h]
        bbox_format (str): The bbox format. Options are ``'xywh'``, ``'xyxy'``
            and ``'center'``.
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        np.ndarray: The flipped bounding boxes.
    """
    direction_options = {"horizontal", "vertical", "diagonal"}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". Options are {direction_options}'
    )

    format_options = {"xywh", "xyxy", "center"}
    assert bbox_format in format_options, (
        f'Invalid bbox format "{bbox_format}". Options are {format_options}'
    )

    bbox_flipped = bbox.copy()
    w, h = image_size

    if direction == "horizontal":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., 0] = w - bbox[..., 0] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[..., ::2] = w - bbox[..., ::2] - 1
    elif direction == "vertical":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., 1] = h - bbox[..., 1] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[..., 1::2] = h - bbox[..., 1::2] - 1
    elif direction == "diagonal":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., :2] = [w, h] - bbox[..., :2] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[...] = [w, h, w, h] - bbox - 1

    return bbox_flipped

def fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
    """Reshape the bbox to a fixed aspect ratio.

    Args:
        bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.darray: The reshaped bbox scales in (n, 2)
    """
    dim = bbox_scale.ndim
    if dim == 1:
        bbox_scale = bbox_scale[None, :]

    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )
    if dim == 1:
        bbox_scale = bbox_scale[0]

    return bbox_scale

def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (
        -0.5 * input_size[0] * math.cos(rot_rad)
        + 0.5 * input_size[1] * math.sin(rot_rad)
        + 0.5 * scale[0]
    )
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (
        -0.5 * input_size[0] * math.sin(rot_rad)
        - 0.5 * input_size[1] * math.cos(rot_rad)
        + 0.5 * scale[1]
    )
    return warp_mat

def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

```

* Файл: sam_3d_body/data/transforms/common.py

```python

from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from sam_3d_body.models.modules import to_2tuple

from .bbox_utils import (
    bbox_xywh2cs,
    bbox_xyxy2cs,
    fix_aspect_ratio,
    get_udp_warp_matrix,
    get_warp_matrix,
)

class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[List[Callable]] = None):
        if transforms is None:
            transforms = []
        else:
            self.transforms = transforms

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)




            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class VisionTransformWrapper:
    """A wrapper to use torchvision transform functions in this codebase."""

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, results: Dict) -> Optional[dict]:
        results["img"] = self.transform(results["img"])
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.transform.__class__.__name__
        return repr_str

class GetBBoxCenterScale(nn.Module):
    """Convert bboxes to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox
        - bbox_format

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

    def forward(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if "bbox_center" in results and "bbox_scale" in results:
            results["bbox_scale"] *= self.padding
        else:
            bbox = results["bbox"]
            bbox_format = results.get("bbox_format", "none")
            if bbox_format == "xywh":
                center, scale = bbox_xywh2cs(bbox, padding=self.padding)
            elif bbox_format == "xyxy":
                center, scale = bbox_xyxy2cs(bbox, padding=self.padding)
            else:
                raise ValueError(
                    "Invalid bbox format: {}".format(results["bbox_format"])
                )

            results["bbox_center"] = center
            results["bbox_scale"] = scale
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f"(padding={self.padding})"
        return repr_str

class SquarePad:
    def __call__(self, results: Dict) -> Optional[dict]:
        assert isinstance(results["img"], Image.Image)
        w, h = results["img"].size

        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)

        results["img"] = F.pad(results["img"], padding, 0, "constant")
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        return repr_str

class ToPIL:
    def __call__(self, results: Dict) -> Optional[dict]:
        if isinstance(results["img"], list):
            if isinstance(results["img"][0], np.ndarray):
                results["img"] = [Image.fromarray(img) for img in results["img"]]
        elif isinstance(results["img"], np.ndarray):
            results["img"] = Image.fromarray(results["img"])

class ToCv2:
    def __call__(self, results: Dict) -> Optional[dict]:
        if isinstance(results["img"], list):
            if isinstance(results["img"][0], Image.Image):
                results["img"] = [np.array(img) for img in results["img"]]
        elif isinstance(results["img"], Image.Image):
            results["img"] = np.array(results["img"])

class TopdownAffine(nn.Module):
    """Get the bbox image as the model input by affine transform.

    Required Keys:
        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints_2d (optional)
        - mask (optional)

    Modified Keys:
        - img
        - bbox_scale

    Added Keys:
        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``
        aspect_ratio (float): both HMR2.0 and Sapiens will expand input bbox to
            a fixed ratio (width/height = 192/256), then expand to the ratio of
            the model input size. E.g., HMR2.0 will eventually expand to 1:1, while
            Sapiens will be 768:1024.

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        use_udp: bool = False,
        aspect_ratio: float = 0.75,
        fix_square: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = to_2tuple(input_size)
        self.use_udp = use_udp
        self.aspect_ratio = aspect_ratio
        self.fix_square = fix_square

    def forward(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """





        w, h = self.input_size
        warp_size = (int(w), int(h))


        results["orig_bbox_scale"] = results["bbox_scale"].copy()
        if self.fix_square and results["bbox_scale"][0] == results["bbox_scale"][1]:

            bbox_scale = fix_aspect_ratio(results["bbox_scale"], aspect_ratio=w / h)
        else:

            bbox_scale = fix_aspect_ratio(
                results["bbox_scale"], aspect_ratio=self.aspect_ratio
            )
            results["bbox_scale"] = fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
        results["bbox_expand_factor"] = (
            results["bbox_scale"].max() / results["orig_bbox_scale"].max()
        )
        rot = 0.0
        if results["bbox_center"].ndim == 2:
            assert results["bbox_center"].shape[0] == 1, (
                "Only support cropping one instance at a time. Got invalid "
                f"shape of bbox_center {results['bbox_center'].shape}."
            )
            center = results["bbox_center"][0]
            scale = results["bbox_scale"][0]
            if "bbox_rotation" in results:
                rot = results["bbox_rotation"][0]
        else:
            center = results["bbox_center"]
            scale = results["bbox_scale"]
            if "bbox_rotation" in results:
                rot = results["bbox_rotation"]

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if "img" not in results:
            pass
        elif isinstance(results["img"], list):
            results["img"] = [
                cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results["img"]
            ]
            height, width = results["img"][0].shape[:2]
            results["ori_img_size"] = np.array([width, height])
        else:
            height, width = results["img"].shape[:2]
            results["ori_img_size"] = np.array([width, height])
            results["img"] = cv2.warpAffine(
                results["img"], warp_mat, warp_size, flags=cv2.INTER_LINEAR
            )

        if results.get("keypoints_2d", None) is not None:
            results["orig_keypoints_2d"] = results["keypoints_2d"].copy()
            transformed_keypoints = results["keypoints_2d"].copy()


            transformed_keypoints[:, :2] = cv2.transform(
                results["keypoints_2d"][None, :, :2], warp_mat
            )[0]
            results["keypoints_2d"] = transformed_keypoints

        if results.get("mask", None) is not None:
            results["mask"] = cv2.warpAffine(
                results["mask"], warp_mat, warp_size, flags=cv2.INTER_LINEAR
            )

        results["img_size"] = np.array([w, h])
        results["input_size"] = np.array([w, h])
        results["affine_trans"] = warp_mat
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(input_size={self.input_size}, "
        repr_str += f"use_udp={self.use_udp})"
        return repr_str

class NormalizeKeypoint(nn.Module):
    """
    Normalize 2D keypoints to range [-0.5, 0.5].

    Required Keys:
        - keypoints_2d
        - img_size

    Modified Keys:
        - keypoints_2d
    """

    def forward(self, results: Dict) -> Optional[dict]:
        if "keypoints_2d" in results:
            img_size = results.get("img_size", results["input_size"])

            results["keypoints_2d"][:, :2] = (
                results["keypoints_2d"][:, :2] / np.array(img_size).reshape(1, 2) - 0.5
            )
        return results

```

* Файл: sam_3d_body/data/transforms/__init__.py

```python

from .bbox_utils import (
    bbox_cs2xywh,
    bbox_cs2xyxy,
    bbox_xywh2cs,
    bbox_xywh2xyxy,
    bbox_xyxy2cs,
    bbox_xyxy2xywh,
    flip_bbox,
    get_udp_warp_matrix,
    get_warp_matrix,
)
from .common import (
    Compose,
    GetBBoxCenterScale,
    NormalizeKeypoint,
    SquarePad,
    TopdownAffine,
    VisionTransformWrapper,
)

```

* Файл: sam_3d_body/data/utils/io.py

```python

import os
import time
from typing import Any, List

import braceexpand
import cv2
import numpy as np

from PIL import Image

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))

def expand_urls(urls: str | List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

def load_image_from_file(
    data_info: dict,
    backend: str = "cv2",
    image_format: str = "rgb",
    retry: int = 10,
) -> dict:
    img = load_image(data_info["img_path"], backend, image_format, retry)
    data_info["img"] = img
    data_info["img_shape"] = img.shape[:2]
    data_info["ori_shape"] = img.shape[:2]
    return data_info

def _pil_load(path: str, image_format: str) -> Image.Image:
    with Image.open(path) as img:
        if img is not None and image_format.lower() == "rgb":
            img = img.convert("RGB")
    return img

def _cv2_load(path: str, image_format: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None and image_format.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_image(
    path: str,
    backend: str = "pil",
    image_format: str = "rgb",
    retry: int = 10,
) -> Any:
    for i_try in range(retry):
        if backend == "pil":
            img = _pil_load(path, image_format)
        elif backend == "cv2":
            img = _cv2_load(path, image_format)
        else:
            raise ValueError("Invalid backend {} for loading image.".format(backend))

        if img is not None:
            return img
        else:
            print("Reading {} failed. Will retry.".format(path))
            time.sleep(1.0)
        if i_try == retry - 1:
            raise Exception("Failed to load image {}".format(path))

def resize_image(img, target_size, center=None, scale=None):
    height, width = img.shape[:2]
    aspect_ratio = width / height


    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size


    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255


    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y : start_y + new_height, start_x : start_x + new_width] = (
        resized_img
    )

    if center is not None and scale is not None:
        ratio_width = new_width / width
        ratio_height = new_height / height

        new_scale = np.stack(
            [scale[:, 0] * ratio_width, scale[:, 1] * ratio_height], axis=1
        )
        new_center = np.stack(
            [center[:, 0] * ratio_width, center[:, 1] * ratio_height], axis=1
        )
        new_center[:, 0] += start_x
        new_center[:, 1] += start_y
    else:
        new_center, new_scale = None, None
    return aspect_ratio, final_img, new_center, new_scale

```

* Файл: sam_3d_body/data/utils/prepare_batch.py

```python

import numpy as np
import torch
from torch.utils.data import default_collate

class NoCollate:
    def __init__(self, data):
        self.data = data

def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]


    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:

        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch

```

* Файл: sam_3d_body/__init__.py

```python

__version__ = "1.0.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]

```

* Файл: sam_3d_body/metadata/__init__.py

```python

import os

OPENPOSE_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

J19_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

LSP_TO_COCO = {
    5: 9,
    6: 8,
    7: 10,
    8: 7,
    9: 11,
    10: 6,
    11: 3,
    12: 2,
    13: 4,
    14: 1,
    15: 5,
    16: 0,
}

OPENPOSE_PERMUTATION = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
J19_PERMUTATION = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
COCO_PERMUTATION = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

MHR70_TO_OPENPOSE = {
    0: 0,
    1: 69,
    2: 6,
    3: 8,
    4: 41,
    5: 5,
    6: 7,
    7: 62,
    9: 10,
    10: 12,
    11: 14,
    12: 9,
    13: 11,
    14: 13,
    15: 2,
    16: 1,
    17: 4,
    18: 3,
    19: 15,
    20: 16,
    21: 17,
    22: 18,
    23: 19,
    24: 20,
}

MHR70_PERMUTATION = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 18, 19, 20, 15, 16, 17, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 64, 63, 66, 65, 68, 67, 69]

MHR70_TO_LSP = {
    0: 14,
    1: 12,
    2: 10,
    3: 9,
    4: 11,
    5: 13,
    6: 41,
    7: 8,
    8: 6,
    9: 5,
    10: 7,
    11: 62,
    12: 69,
}

```

* Файл: sam_3d_body/metadata/mhr70.py

```python

"""The first 70 of 308 MHR keypoints, ignoring the rest for face keypoints"""

mhr_names = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip",
    "right-middle-first-joint",
    "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip",
    "right-ring-first-joint",
    "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip",
    "right-pinky-first-joint",
    "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip",
    "left-thumb-first-joint",
    "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip",
    "left-index-first-joint",
    "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip",
    "left-middle-first-joint",
    "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip",
    "left-ring-first-joint",
    "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip",
    "left-pinky-first-joint",
    "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
]

pose_info = dict(
    pose_format="mhr70",
    paper_info=dict(
        author="",
        year="",
        homepage="",
    ),
    min_visible_keypoints=8,
    image_height=4096,
    image_width=2668,
    original_keypoint_info={
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_hip",
        10: "right_hip",
        11: "left_knee",
        12: "right_knee",
        13: "left_ankle",
        14: "right_ankle",
        15: "left_big_toe_tip",
        16: "left_small_toe_tip",
        17: "left_heel",
        18: "right_big_toe_tip",
        19: "right_small_toe_tip",
        20: "right_heel",
        21: "right_thumb_tip",
        22: "right_thumb_first_joint",
        23: "right_thumb_second_joint",
        24: "right_thumb_third_joint",
        25: "right_index_tip",
        26: "right_index_first_joint",
        27: "right_index_second_joint",
        28: "right_index_third_joint",
        29: "right_middle_tip",
        30: "right_middle_first_joint",
        31: "right_middle_second_joint",
        32: "right_middle_third_joint",
        33: "right_ring_tip",
        34: "right_ring_first_joint",
        35: "right_ring_second_joint",
        36: "right_ring_third_joint",
        37: "right_pinky_tip",
        38: "right_pinky_first_joint",
        39: "right_pinky_second_joint",
        40: "right_pinky_third_joint",
        41: "right_wrist",
        42: "left_thumb_tip",
        43: "left_thumb_first_joint",
        44: "left_thumb_second_joint",
        45: "left_thumb_third_joint",
        46: "left_index_tip",
        47: "left_index_first_joint",
        48: "left_index_second_joint",
        49: "left_index_third_joint",
        50: "left_middle_tip",
        51: "left_middle_first_joint",
        52: "left_middle_second_joint",
        53: "left_middle_third_joint",
        54: "left_ring_tip",
        55: "left_ring_first_joint",
        56: "left_ring_second_joint",
        57: "left_ring_third_joint",
        58: "left_pinky_tip",
        59: "left_pinky_first_joint",
        60: "left_pinky_second_joint",
        61: "left_pinky_third_joint",
        62: "left_wrist",
        63: "left_olecranon",
        64: "right_olecranon",
        65: "left_cubital_fossa",
        66: "right_cubital_fossa",
        67: "left_acromion",
        68: "right_acromion",
        69: "neck",
    },
    keypoint_info={
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(
            name="left_eye", id=1, color=[51, 153, 255], type="upper", swap="right_eye"
        ),
        2: dict(
            name="right_eye", id=2, color=[51, 153, 255], type="upper", swap="left_eye"
        ),
        3: dict(
            name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"
        ),
        4: dict(
            name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"
        ),
        5: dict(
            name="left_shoulder",
            id=5,
            color=[51, 153, 255],
            type="upper",
            swap="right_shoulder",
        ),
        6: dict(
            name="right_shoulder",
            id=6,
            color=[51, 153, 255],
            type="upper",
            swap="left_shoulder",
        ),
        7: dict(
            name="left_elbow",
            id=7,
            color=[51, 153, 255],
            type="upper",
            swap="right_elbow",
        ),
        8: dict(
            name="right_elbow",
            id=8,
            color=[51, 153, 255],
            type="upper",
            swap="left_elbow",
        ),
        9: dict(
            name="left_hip", id=9, color=[51, 153, 255], type="lower", swap="right_hip"
        ),
        10: dict(
            name="right_hip", id=10, color=[51, 153, 255], type="lower", swap="left_hip"
        ),
        11: dict(
            name="left_knee",
            id=11,
            color=[51, 153, 255],
            type="lower",
            swap="right_knee",
        ),
        12: dict(
            name="right_knee",
            id=12,
            color=[51, 153, 255],
            type="lower",
            swap="left_knee",
        ),
        13: dict(
            name="left_ankle",
            id=13,
            color=[51, 153, 255],
            type="lower",
            swap="right_ankle",
        ),
        14: dict(
            name="right_ankle",
            id=14,
            color=[51, 153, 255],
            type="lower",
            swap="left_ankle",
        ),
        15: dict(
            name="left_big_toe",
            id=15,
            color=[51, 153, 255],
            type="lower",
            swap="right_big_toe",
        ),
        16: dict(
            name="left_small_toe",
            id=16,
            color=[51, 153, 255],
            type="lower",
            swap="right_small_toe",
        ),
        17: dict(
            name="left_heel",
            id=17,
            color=[51, 153, 255],
            type="lower",
            swap="right_heel",
        ),
        18: dict(
            name="right_big_toe",
            id=18,
            color=[51, 153, 255],
            type="lower",
            swap="left_big_toe",
        ),
        19: dict(
            name="right_small_toe",
            id=19,
            color=[51, 153, 255],
            type="lower",
            swap="left_small_toe",
        ),
        20: dict(
            name="right_heel",
            id=20,
            color=[51, 153, 255],
            type="lower",
            swap="left_heel",
        ),
        21: dict(
            name="right_thumb4",
            id=21,
            color=[51, 153, 255],
            type="upper",
            swap="left_thumb4",
        ),
        22: dict(
            name="right_thumb3",
            id=22,
            color=[51, 153, 255],
            type="upper",
            swap="left_thumb3",
        ),
        23: dict(
            name="right_thumb2",
            id=23,
            color=[51, 153, 255],
            type="upper",
            swap="left_thumb2",
        ),
        24: dict(
            name="right_thumb_third_joint",
            id=24,
            color=[51, 153, 255],
            type="upper",
            swap="left_thumb_third_joint",
        ),
        25: dict(
            name="right_forefinger4",
            id=25,
            color=[51, 153, 255],
            type="upper",
            swap="left_forefinger4",
        ),
        26: dict(
            name="right_forefinger3",
            id=26,
            color=[51, 153, 255],
            type="upper",
            swap="left_forefinger3",
        ),
        27: dict(
            name="right_forefinger2",
            id=27,
            color=[51, 153, 255],
            type="upper",
            swap="left_forefinger2",
        ),
        28: dict(
            name="right_forefinger_third_joint",
            id=28,
            color=[51, 153, 255],
            type="upper",
            swap="left_forefinger_third_joint",
        ),
        29: dict(
            name="right_middle_finger4",
            id=29,
            color=[51, 153, 255],
            type="upper",
            swap="left_middle_finger4",
        ),
        30: dict(
            name="right_middle_finger3",
            id=30,
            color=[51, 153, 255],
            type="upper",
            swap="left_middle_finger3",
        ),
        31: dict(
            name="right_middle_finger2",
            id=31,
            color=[51, 153, 255],
            type="upper",
            swap="left_middle_finger2",
        ),
        32: dict(
            name="right_middle_finger_third_joint",
            id=32,
            color=[51, 153, 255],
            type="upper",
            swap="left_middle_finger_third_joint",
        ),
        33: dict(
            name="right_ring_finger4",
            id=33,
            color=[51, 153, 255],
            type="upper",
            swap="left_ring_finger4",
        ),
        34: dict(
            name="right_ring_finger3",
            id=34,
            color=[51, 153, 255],
            type="upper",
            swap="left_ring_finger3",
        ),
        35: dict(
            name="right_ring_finger2",
            id=35,
            color=[51, 153, 255],
            type="upper",
            swap="left_ring_finger2",
        ),
        36: dict(
            name="right_ring_finger_third_joint",
            id=36,
            color=[51, 153, 255],
            type="upper",
            swap="left_ring_finger_third_joint",
        ),
        37: dict(
            name="right_pinky_finger4",
            id=37,
            color=[51, 153, 255],
            type="upper",
            swap="left_pinky_finger4",
        ),
        38: dict(
            name="right_pinky_finger3",
            id=38,
            color=[51, 153, 255],
            type="upper",
            swap="left_pinky_finger3",
        ),
        39: dict(
            name="right_pinky_finger2",
            id=39,
            color=[51, 153, 255],
            type="upper",
            swap="left_pinky_finger2",
        ),
        40: dict(
            name="right_pinky_finger_third_joint",
            id=40,
            color=[51, 153, 255],
            type="upper",
            swap="left_pinky_finger_third_joint",
        ),
        41: dict(
            name="right_wrist",
            id=41,
            color=[51, 153, 255],
            type="upper",
            swap="left_wrist",
        ),
        42: dict(
            name="left_thumb4",
            id=42,
            color=[51, 153, 255],
            type="upper",
            swap="right_thumb4",
        ),
        43: dict(
            name="left_thumb3",
            id=43,
            color=[51, 153, 255],
            type="upper",
            swap="right_thumb3",
        ),
        44: dict(
            name="left_thumb2",
            id=44,
            color=[51, 153, 255],
            type="upper",
            swap="right_thumb2",
        ),
        45: dict(
            name="left_thumb_third_joint",
            id=45,
            color=[51, 153, 255],
            type="upper",
            swap="right_thumb_third_joint",
        ),  * doesnt match with wholebody
        46: dict(
            name="left_forefinger4",
            id=46,
            color=[51, 153, 255],
            type="upper",
            swap="right_forefinger4",
        ),
        47: dict(
            name="left_forefinger3",
            id=47,
            color=[51, 153, 255],
            type="upper",
            swap="right_forefinger3",
        ),
        48: dict(
            name="left_forefinger2",
            id=48,
            color=[51, 153, 255],
            type="upper",
            swap="right_forefinger2",
        ),
        49: dict(
            name="left_forefinger_third_joint",
            id=49,
            color=[51, 153, 255],
            type="upper",
            swap="right_forefinger_third_joint",
        ),
        50: dict(
            name="left_middle_finger4",
            id=50,
            color=[51, 153, 255],
            type="upper",
            swap="right_middle_finger4",
        ),
        51: dict(
            name="left_middle_finger3",
            id=51,
            color=[51, 153, 255],
            type="upper",
            swap="right_middle_finger3",
        ),
        52: dict(
            name="left_middle_finger2",
            id=52,
            color=[51, 153, 255],
            type="upper",
            swap="right_middle_finger2",
        ),
        53: dict(
            name="left_middle_finger_third_joint",
            id=53,
            color=[51, 153, 255],
            type="upper",
            swap="right_middle_finger_third_joint",
        ),
        54: dict(
            name="left_ring_finger4",
            id=54,
            color=[51, 153, 255],
            type="upper",
            swap="right_ring_finger4",
        ),
        55: dict(
            name="left_ring_finger3",
            id=55,
            color=[51, 153, 255],
            type="upper",
            swap="right_ring_finger3",
        ),
        56: dict(
            name="left_ring_finger2",
            id=56,
            color=[51, 153, 255],
            type="upper",
            swap="right_ring_finger2",
        ),
        57: dict(
            name="left_ring_finger_third_joint",
            id=57,
            color=[51, 153, 255],
            type="upper",
            swap="right_ring_finger_third_joint",
        ),
        58: dict(
            name="left_pinky_finger4",
            id=58,
            color=[51, 153, 255],
            type="upper",
            swap="right_pinky_finger4",
        ),
        59: dict(
            name="left_pinky_finger3",
            id=59,
            color=[51, 153, 255],
            type="upper",
            swap="right_pinky_finger3",
        ),
        60: dict(
            name="left_pinky_finger2",
            id=60,
            color=[51, 153, 255],
            type="upper",
            swap="right_pinky_finger2",
        ),
        61: dict(
            name="left_pinky_finger_third_joint",
            id=61,
            color=[51, 153, 255],
            type="upper",
            swap="right_pinky_finger_third_joint",
        ),
        62: dict(
            name="left_wrist",
            id=62,
            color=[51, 153, 255],
            type="upper",
            swap="right_wrist",
        ),
        63: dict(
            name="left_olecranon",
            id=63,
            color=[51, 153, 255],
            type="",
            swap="right_olecranon",
        ),
        64: dict(
            name="right_olecranon",
            id=64,
            color=[51, 153, 255],
            type="",
            swap="left_olecranon",
        ),
        65: dict(
            name="left_cubital_fossa",
            id=65,
            color=[51, 153, 255],
            type="",
            swap="right_cubital_fossa",
        ),
        66: dict(
            name="right_cubital_fossa",
            id=66,
            color=[51, 153, 255],
            type="",
            swap="left_cubital_fossa",
        ),
        67: dict(
            name="left_acromion",
            id=67,
            color=[51, 153, 255],
            type="",
            swap="right_acromion",
        ),
        68: dict(
            name="right_acromion",
            id=68,
            color=[51, 153, 255],
            type="",
            swap="left_acromion",
        ),
        69: dict(name="neck", id=69, color=[51, 153, 255], type="", swap=""),
    },
    skeleton_info={
        0: dict(link=("left_ankle", "left_knee"), id=0, color=[0, 255, 0]),
        1: dict(link=("left_knee", "left_hip"), id=1, color=[0, 255, 0]),
        2: dict(link=("right_ankle", "right_knee"), id=2, color=[255, 128, 0]),
        3: dict(link=("right_knee", "right_hip"), id=3, color=[255, 128, 0]),
        4: dict(link=("left_hip", "right_hip"), id=4, color=[51, 153, 255]),
        5: dict(link=("left_shoulder", "left_hip"), id=5, color=[51, 153, 255]),
        6: dict(link=("right_shoulder", "right_hip"), id=6, color=[51, 153, 255]),
        7: dict(link=("left_shoulder", "right_shoulder"), id=7, color=[51, 153, 255]),
        8: dict(link=("left_shoulder", "left_elbow"), id=8, color=[0, 255, 0]),
        9: dict(link=("right_shoulder", "right_elbow"), id=9, color=[255, 128, 0]),
        10: dict(link=("left_elbow", "left_wrist"), id=10, color=[0, 255, 0]),
        11: dict(link=("right_elbow", "right_wrist"), id=11, color=[255, 128, 0]),
        12: dict(link=("left_eye", "right_eye"), id=12, color=[51, 153, 255]),
        13: dict(link=("nose", "left_eye"), id=13, color=[51, 153, 255]),
        14: dict(link=("nose", "right_eye"), id=14, color=[51, 153, 255]),
        15: dict(link=("left_eye", "left_ear"), id=15, color=[51, 153, 255]),
        16: dict(link=("right_eye", "right_ear"), id=16, color=[51, 153, 255]),
        17: dict(link=("left_ear", "left_shoulder"), id=17, color=[51, 153, 255]),
        18: dict(link=("right_ear", "right_shoulder"), id=18, color=[51, 153, 255]),
        19: dict(link=("left_ankle", "left_big_toe"), id=19, color=[0, 255, 0]),
        20: dict(link=("left_ankle", "left_small_toe"), id=20, color=[0, 255, 0]),
        21: dict(link=("left_ankle", "left_heel"), id=21, color=[0, 255, 0]),
        22: dict(link=("right_ankle", "right_big_toe"), id=22, color=[255, 128, 0]),
        23: dict(link=("right_ankle", "right_small_toe"), id=23, color=[255, 128, 0]),
        24: dict(link=("right_ankle", "right_heel"), id=24, color=[255, 128, 0]),
        25: dict(
            link=("left_wrist", "left_thumb_third_joint"), id=25, color=[255, 128, 0]
        ),
        26: dict(
            link=("left_thumb_third_joint", "left_thumb2"), id=26, color=[255, 128, 0]
        ),
        27: dict(link=("left_thumb2", "left_thumb3"), id=27, color=[255, 128, 0]),
        28: dict(link=("left_thumb3", "left_thumb4"), id=28, color=[255, 128, 0]),
        29: dict(
            link=("left_wrist", "left_forefinger_third_joint"),
            id=29,
            color=[255, 153, 255],
        ),
        30: dict(
            link=("left_forefinger_third_joint", "left_forefinger2"),
            id=30,
            color=[255, 153, 255],
        ),
        31: dict(
            link=("left_forefinger2", "left_forefinger3"), id=31, color=[255, 153, 255]
        ),
        32: dict(
            link=("left_forefinger3", "left_forefinger4"), id=32, color=[255, 153, 255]
        ),
        33: dict(
            link=("left_wrist", "left_middle_finger_third_joint"),
            id=33,
            color=[102, 178, 255],
        ),
        34: dict(
            link=("left_middle_finger_third_joint", "left_middle_finger2"),
            id=34,
            color=[102, 178, 255],
        ),
        35: dict(
            link=("left_middle_finger2", "left_middle_finger3"),
            id=35,
            color=[102, 178, 255],
        ),
        36: dict(
            link=("left_middle_finger3", "left_middle_finger4"),
            id=36,
            color=[102, 178, 255],
        ),
        37: dict(
            link=("left_wrist", "left_ring_finger_third_joint"),
            id=37,
            color=[255, 51, 51],
        ),
        38: dict(
            link=("left_ring_finger_third_joint", "left_ring_finger2"),
            id=38,
            color=[255, 51, 51],
        ),
        39: dict(
            link=("left_ring_finger2", "left_ring_finger3"), id=39, color=[255, 51, 51]
        ),
        40: dict(
            link=("left_ring_finger3", "left_ring_finger4"), id=40, color=[255, 51, 51]
        ),
        41: dict(
            link=("left_wrist", "left_pinky_finger_third_joint"),
            id=41,
            color=[0, 255, 0],
        ),
        42: dict(
            link=("left_pinky_finger_third_joint", "left_pinky_finger2"),
            id=42,
            color=[0, 255, 0],
        ),
        43: dict(
            link=("left_pinky_finger2", "left_pinky_finger3"), id=43, color=[0, 255, 0]
        ),
        44: dict(
            link=("left_pinky_finger3", "left_pinky_finger4"), id=44, color=[0, 255, 0]
        ),
        45: dict(
            link=("right_wrist", "right_thumb_third_joint"), id=45, color=[255, 128, 0]
        ),
        46: dict(
            link=("right_thumb_third_joint", "right_thumb2"), id=46, color=[255, 128, 0]
        ),
        47: dict(link=("right_thumb2", "right_thumb3"), id=47, color=[255, 128, 0]),
        48: dict(link=("right_thumb3", "right_thumb4"), id=48, color=[255, 128, 0]),
        49: dict(
            link=("right_wrist", "right_forefinger_third_joint"),
            id=49,
            color=[255, 153, 255],
        ),
        50: dict(
            link=("right_forefinger_third_joint", "right_forefinger2"),
            id=50,
            color=[255, 153, 255],
        ),
        51: dict(
            link=("right_forefinger2", "right_forefinger3"),
            id=51,
            color=[255, 153, 255],
        ),
        52: dict(
            link=("right_forefinger3", "right_forefinger4"),
            id=52,
            color=[255, 153, 255],
        ),
        53: dict(
            link=("right_wrist", "right_middle_finger_third_joint"),
            id=53,
            color=[102, 178, 255],
        ),
        54: dict(
            link=("right_middle_finger_third_joint", "right_middle_finger2"),
            id=54,
            color=[102, 178, 255],
        ),
        55: dict(
            link=("right_middle_finger2", "right_middle_finger3"),
            id=55,
            color=[102, 178, 255],
        ),
        56: dict(
            link=("right_middle_finger3", "right_middle_finger4"),
            id=56,
            color=[102, 178, 255],
        ),
        57: dict(
            link=("right_wrist", "right_ring_finger_third_joint"),
            id=57,
            color=[255, 51, 51],
        ),
        58: dict(
            link=("right_ring_finger_third_joint", "right_ring_finger2"),
            id=58,
            color=[255, 51, 51],
        ),
        59: dict(
            link=("right_ring_finger2", "right_ring_finger3"),
            id=59,
            color=[255, 51, 51],
        ),
        60: dict(
            link=("right_ring_finger3", "right_ring_finger4"),
            id=60,
            color=[255, 51, 51],
        ),
        61: dict(
            link=("right_wrist", "right_pinky_finger_third_joint"),
            id=61,
            color=[0, 255, 0],
        ),
        62: dict(
            link=("right_pinky_finger_third_joint", "right_pinky_finger2"),
            id=62,
            color=[0, 255, 0],
        ),
        63: dict(
            link=("right_pinky_finger2", "right_pinky_finger3"),
            id=63,
            color=[0, 255, 0],
        ),
        64: dict(
            link=("right_pinky_finger3", "right_pinky_finger4"),
            id=64,
            color=[0, 255, 0],
        ),
    },
    joint_weights=[1.0] * 70,
    body_keypoint_names=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    foot_keypoint_names=[
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ],
    left_hand_keypoint_names=[
        "left_thumb4",
        "left_thumb3",
        "left_thumb2",
        "left_thumb_third_joint",
        "left_forefinger4",
        "left_forefinger3",
        "left_forefinger2",
        "left_forefinger_third_joint",
        "left_middle_finger4",
        "left_middle_finger3",
        "left_middle_finger2",
        "left_middle_finger_third_joint",
        "left_ring_finger4",
        "left_ring_finger3",
        "left_ring_finger2",
        "left_ring_finger_third_joint",
        "left_pinky_finger4",
        "left_pinky_finger3",
        "left_pinky_finger2",
        "left_pinky_finger_third_joint",
    ],
    right_hand_keypoint_names=[
        "right_thumb4",
        "right_thumb3",
        "right_thumb2",
        "right_thumb_third_joint",
        "right_forefinger4",
        "right_forefinger3",
        "right_forefinger2",
        "right_forefinger_third_joint",
        "right_middle_finger4",
        "right_middle_finger3",
        "right_middle_finger2",
        "right_middle_finger_third_joint",
        "right_ring_finger4",
        "right_ring_finger3",
        "right_ring_finger2",
        "right_ring_finger_third_joint",
        "right_pinky_finger4",
        "right_pinky_finger3",
        "right_pinky_finger2",
        "right_pinky_finger_third_joint",
    ],
    * 7 of them
    extra_keypoint_names=[
        "neck",
        "left_olecranon",
        "right_olecranon",
        "left_cubital_fossa",
        "right_cubital_fossa",
        "left_acromion",
        "right_acromion",
    ],
    sigmas=[],
)

```

* Файл: sam_3d_body/models/backbones/dinov3.py

```python

import torch
from torch import nn

class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        self.encoder = torch.hub.load(
            "facebookresearch/dinov3",
            self.name,
            source="github",
            pretrained=False,
            drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):

            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers

```

* Файл: sam_3d_body/models/backbones/__init__.py

```python

def create_backbone(name, cfg=None):
    if name in ["vit_hmr"]:
        from .vit import vit

        backbone = vit(cfg)
    elif name in ["vit_hmr_512_384"]:
        from .vit import vit512_384

        backbone = vit512_384(cfg)
    elif name in ["vit_l"]:
        from .vit import vit_l

        backbone = vit_l(cfg)
    elif name in ["vit_b"]:
        from .vit import vit_b

        backbone = vit_b(cfg)
    elif name in [
        "dinov3_vit7b",
        "dinov3_vith16plus",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
    ]:
        from .dinov3 import Dinov3Backbone

        backbone = Dinov3Backbone(name, cfg=cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")

    return backbone

```

* Файл: sam_3d_body/models/backbones/vit.py

```python

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except:
    print("No Flash Attention!")

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from ..modules.transformer import LayerNorm32

def vit(cfg):
    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        norm_layer=LayerNorm32,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        flash_attn=cfg.MODEL.BACKBONE.get("FLASH_ATTN", False),
    )

def vit_l(cfg):
    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ratio=1,
        norm_layer=LayerNorm32,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        flash_attn=cfg.MODEL.BACKBONE.get("FLASH_ATTN", False),
    )

def vit_b(cfg):
    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        norm_layer=LayerNorm32,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        flash_attn=cfg.MODEL.BACKBONE.get("FLASH_ATTN", False),
    )

def vit256(cfg):
    return ViT(
        img_size=(256, 256),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        norm_layer=LayerNorm32,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        flash_attn=cfg.MODEL.BACKBONE.get("FLASH_ATTN", False),
    )

def vit512_384(cfg):
    return ViT(
        img_size=(512, 384),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        norm_layer=LayerNorm32,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        flash_attn=cfg.MODEL.BACKBONE.get("FLASH_ATTN", False),
    )

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = (
            F.interpolate(
                abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(B, -1, C)
        )

    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class FlashAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or (dim // num_heads)
        self.head_dim = head_dim
        self.dim = dim
        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()


        if q.dtype == torch.float32:
            q = q.half()
            k = k.half()
            v = v.half()

        out = flash_attn_func(
            q, k, v, dropout_p=self.attn_drop, causal=False
        )


        out = out.reshape(B, N, -1)
        out = out.to(x.dtype)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        flash_attn=False,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if flash_attn:
            self.attn = FlashAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )


        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        )
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class PatchEmbedNoPadding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        )
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=0,
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        use_checkpoint=False,
        frozen_stages=-1,
        ratio=1,
        last_norm=True,
        patch_padding="pad",
        freeze_attn=False,
        freeze_ffn=False,
        flash_attn=False,
        no_patch_padding=False,
    ):

        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = self.embed_dims = (
            embed_dim
        )
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            if no_patch_padding:
                self.patch_embed = PatchEmbedNoPadding(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    ratio=ratio,
                )
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    ratio=ratio,
                )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    flash_attn=flash_attn,
                )
                for i in range(depth)
            ]
        )

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, extra_embed=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:


            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        if extra_embed is not None:
            x = x + extra_embed.flatten(2).transpose(1, 2).to(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x, *args, **kwargs):
        x = self.forward_features(x, *args, **kwargs)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

```

* Файл: sam_3d_body/models/decoders/__init__.py

```python

from .keypoint_prompt_sampler import build_keypoint_sampler
from .prompt_encoder import PromptEncoder
from .promptable_decoder import PromptableDecoder

def build_decoder(cfg, context_dim=None):
    if cfg.TYPE == "sam":
        return PromptableDecoder(
            dims=cfg.DIM,
            context_dims=context_dim,
            depth=cfg.DEPTH,
            num_heads=cfg.HEADS,
            head_dims=cfg.DIM_HEAD,
            mlp_dims=cfg.MLP_DIM,
            layer_scale_init_value=cfg.LAYER_SCALE_INIT,
            drop_rate=cfg.DROP_RATE,
            attn_drop_rate=cfg.ATTN_DROP_RATE,
            drop_path_rate=cfg.DROP_PATH_RATE,
            ffn_type=cfg.FFN_TYPE,
            enable_twoway=cfg.ENABLE_TWOWAY,
            repeat_pe=cfg.REPEAT_PE,
            frozen=cfg.get("FROZEN", False),
            do_interm_preds=cfg.get("DO_INTERM_PREDS", False),
            do_keypoint_tokens=cfg.get("DO_KEYPOINT_TOKENS", False),
            keypoint_token_update=cfg.get("KEYPOINT_TOKEN_UPDATE", None),
        )
    else:
        raise ValueError("Invalid decoder type: ", cfg.TYPE)

```

* Файл: sam_3d_body/models/decoders/keypoint_prompt_sampler.py

```python

import random
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from omegaconf import DictConfig

def build_keypoint_sampler(sampler_cfg, prompt_keypoints, keybody_idx):
    sampler_type = sampler_cfg.get("TYPE", "v1")
    if sampler_type == "v1":
        sampler_cls = KeypointSamplerV1
    else:
        raise ValueError("Invalid sampler type: ", sampler_type)

    return sampler_cls(sampler_cfg, prompt_keypoints, keybody_idx)

class BaseKeypointSampler(ABC):
    @abstractmethod
    def sample(
        self, gt_keypoints: torch.Tensor, pred_keypoints: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        pass

    def _get_worst_keypoint(self, distances, keypoint_list):

        cur_dist = torch.ones_like(distances) * -1
        cur_dist[keypoint_list] = distances[keypoint_list]
        keypoint_idx = int(cur_dist.argmax())
        if cur_dist[keypoint_idx] > self.distance_thresh:
            valid_keypoint = True
        else:
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _get_random_keypoint(self, distances, keypoint_list):
        candidates = [idx for idx in keypoint_list if distances[idx] > 0]
        if len(candidates):
            keypoint_idx = random.choice(candidates)
            valid_keypoint = True
        else:
            keypoint_idx = None
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _masked_distance(self, x, y, mask=None):
        """
        Args:
            x, y: [B, K, D]
            mask: [B, K]
        Return:
            distances: [K, B]
        """
        distances = (x - y).pow(2).sum(dim=-1)
        if mask is not None:
            distances[mask] = -1
        return distances.T

class KeypointSamplerV1(BaseKeypointSampler):
    def __init__(
        self,
        sampler_cfg: DictConfig,
        prompt_keypoints: Dict,
        keybody_idx: List,
    ):
        self.prompt_keypoints = prompt_keypoints
        self._keybody_idx = keybody_idx
        self._non_keybody_idx = [
            idx for idx in self.prompt_keypoints if idx not in self._keybody_idx
        ]

        self.keybody_ratio = sampler_cfg.get("KEYBODY_RATIO", 0.8)
        self.worst_ratio = sampler_cfg.get("WORST_RATIO", 0.8)
        self.negative_ratio = sampler_cfg.get("NEGATIVE_RATIO", 0.0)
        self.dummy_ratio = sampler_cfg.get("DUMMY_RATIO", 0.1)
        self.distance_thresh = sampler_cfg.get("DISTANCE_THRESH", 0.0)

    def sample(
        self,
        gt_keypoints_2d: torch.Tensor,
        pred_keypoints_2d: torch.Tensor,
        is_train: bool = True,
        force_dummy: bool = False,
    ) -> torch.Tensor:



        mask_1 = gt_keypoints_2d[:, :, -1] < 0.5
        mask_2 = (
            (gt_keypoints_2d[:, :, :2] > 0.5) | (gt_keypoints_2d[:, :, :2] < -0.5)
        ).any(dim=-1)


        if not is_train or torch.rand(1).item() > self.negative_ratio:
            mask = mask_1 | mask_2

        else:
            mask_3 = (
                (pred_keypoints_2d[:, :, :2] > 0.5)
                | (pred_keypoints_2d[:, :, :2] < -0.5)
            ).any(dim=-1)

            mask = mask_1 | (mask_2 & mask_3)



        distances = self._masked_distance(
            pred_keypoints_2d, gt_keypoints_2d[..., :2], mask
        )

        batch_size = distances.shape[1]
        keypoints_prompt = []
        for b in range(batch_size):



            if not is_train or torch.rand(1).item() < self.worst_ratio:
                sampler = self._get_worst_keypoint

            else:
                sampler = self._get_random_keypoint



            if not is_train or torch.rand(1).item() < self.keybody_ratio:
                cur_idx = self._keybody_idx
                alt_idx = self._non_keybody_idx

            else:
                cur_idx = self._non_keybody_idx
                alt_idx = self._keybody_idx



            if not is_train or torch.rand(1).item() > self.dummy_ratio:
                keypoint_idx, valid_keypoint = sampler(distances[:, b], cur_idx)

                if not valid_keypoint:

                    keypoint_idx, valid_keypoint = self._get_worst_keypoint(
                        distances[:, b], alt_idx
                    )
            else:
                valid_keypoint = False

            if valid_keypoint:
                cur_point = gt_keypoints_2d[b, keypoint_idx].clone()
                if torch.any(cur_point[:2] > 0.5) or torch.any(cur_point[:2] < -0.5):

                    cur_point[:2] = pred_keypoints_2d[b, keypoint_idx][:2]
                    cur_point = torch.clamp(
                        cur_point + 0.5, min=0.0, max=1.0
                    )
                    cur_point[-1] = -1

                else:
                    cur_point = torch.clamp(
                        cur_point + 0.5, min=0.0, max=1.0
                    )
                    cur_point[-1] = self.prompt_keypoints[
                        keypoint_idx
                    ]

            else:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2


            if force_dummy:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2

            keypoints_prompt.append(cur_point)


        keypoints_prompt = torch.stack(keypoints_prompt, dim=0).view(batch_size, 1, 3)
        return keypoints_prompt

```

* Файл: sam_3d_body/models/decoders/promptable_decoder.py

```python

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..modules.transformer import build_norm_layer, TransformerDecoderLayer

class PromptableDecoder(nn.Module):
    """Cross-attention based Transformer decoder with prompts input.

    Args:
        token_dims (int): The dimension of input pose tokens.
        prompt_dims (int): The dimension of input prompt tokens.
        context_dims (int): The dimension of image context features.
        dims (int): The projected dimension of all tokens in the decoder.
        depth (int): The number of layers for Transformer decoder.
        num_heads (int): The number of heads for multi-head attention.
        head_dims (int): The dimension of each head.
        mlp_dims (int): The dimension of hidden layers in MLP.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        enable_twoway (bool): Whether to enable two-way Transformer (used in SAM).
        repeat_pe (bool): Whether to re-add PE at each layer (used in SAM)
    """

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        frozen: bool = False,
        do_interm_preds: bool = False,
        do_keypoint_tokens: bool = False,
        keypoint_token_update: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TransformerDecoderLayer(
                    token_dims=dims,
                    context_dims=context_dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    mlp_dims=mlp_dims,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    ffn_type=ffn_type,
                    act_layer=act_layer,
                    norm_cfg=norm_cfg,
                    enable_twoway=enable_twoway,
                    repeat_pe=repeat_pe,
                    skip_first_pe=(i == 0),
                )
            )

        self.norm_final = build_norm_layer(norm_cfg, dims)
        self.do_interm_preds = do_interm_preds
        self.do_keypoint_tokens = do_keypoint_tokens
        self.keypoint_token_update = keypoint_token_update

        self.frozen = frozen
        self._freeze_stages()

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        channel_first: bool = True,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
    ):
        """
        Args:
            token_embedding: [B, N, C]
            image_embedding: [B, C, H, W]
        """
        if channel_first:
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            if image_augment is not None:
                image_augment = image_augment.flatten(2).permute(0, 2, 1)
            if hand_embeddings is not None:
                hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
                hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
                if len(hand_augment) == 1:

                    assert len(hand_augment.shape) == 3
                    hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        if self.do_interm_preds:
            assert token_to_pose_output_fn is not None
            all_pose_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            if hand_embeddings is None:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    image_embedding,
                    token_augment,
                    image_augment,
                    token_mask,
                )
            else:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    torch.cat([image_embedding, hand_embeddings], dim=1),
                    token_augment,
                    torch.cat([image_augment, hand_augment], dim=1),
                    token_mask,
                )
                image_embedding = image_embedding[:, : image_augment.shape[1]]

            if self.do_interm_preds and layer_idx < len(self.layers) - 1:
                curr_pose_output = token_to_pose_output_fn(
                    self.norm_final(token_embedding),
                    prev_pose_output=(
                        all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                    ),
                    layer_idx=layer_idx,
                )
                all_pose_outputs.append(curr_pose_output)

                if self.keypoint_token_update:
                    assert keypoint_token_update_fn is not None
                    token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                        token_embedding, token_augment, curr_pose_output, layer_idx
                    )

        out = self.norm_final(token_embedding)

        if self.do_interm_preds:
            curr_pose_output = token_to_pose_output_fn(
                out,
                prev_pose_output=(
                    all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                ),
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)

            return out, all_pose_outputs
        else:
            return out

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for layer in self.layers:
                layer.eval()
            self.norm_final.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Convert the model into training mode.
        (not called by lightning in trainer.fit() actually)
        """
        super().train(mode)
        self._freeze_stages()

```

* Файл: sam_3d_body/models/decoders/prompt_encoder.py

```python

from typing import Any, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from sam_3d_body.models.modules.transformer import LayerNorm2d

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_body_joints: int,


        frozen: bool = False,
        mask_embed_type: Optional[str] = None,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
            embed_dim (int): The prompts' embedding dimension
            num_body_joints (int): The number of body joints
            img_size (Tuple): The padded size of the image as input
                to the image encoder, as (H, W).
            patch_resolution (Tuple): image path size, as (H, W)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_body_joints = num_body_joints




        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(self.num_body_joints)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.invalid_point_embed = nn.Embedding(1, embed_dim)


        if mask_embed_type in ["v1"]:
            mask_in_chans = 16
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=4, stride=4),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
        elif mask_embed_type in ["v2"]:
            mask_in_chans = 256
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 64, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 64),
                nn.GELU(),
                nn.Conv2d(
                    mask_in_chans // 64,
                    mask_in_chans // 16,
                    kernel_size=2,
                    stride=2,
                ),
                LayerNorm2d(mask_in_chans // 16),
                nn.GELU(),
                nn.Conv2d(
                    mask_in_chans // 16, mask_in_chans // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
        else:
            assert mask_embed_type is None

        if mask_embed_type is not None:

            nn.init.zeros_(self.mask_downscaling[-1].weight)
            nn.init.zeros_(self.mask_downscaling[-1].bias)

            self.no_mask_embed = nn.Embedding(1, embed_dim)
            nn.init.zeros_(self.no_mask_embed.weight)

        self.frozen = frozen
        self._freeze_stages()

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(size).unsqueeze(0)

    def _embed_keypoints(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embeds point prompts.
        Assuming points have been normalized to [0, 1].

        Output shape [B, N, C], mask shape [B, N]
        """
        assert points.min() >= 0 and points.max() <= 1
        point_embedding = self.pe_layer._pe_encoding(points.to(torch.float))
        point_embedding[labels == -2] = 0.0
        point_embedding[labels == -2] += self.invalid_point_embed.weight
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        for i in range(self.num_body_joints):
            point_embedding[labels == i] += self.point_embeddings[i].weight

        point_mask = labels > -2
        return point_embedding, point_mask

    def _get_batch_size(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if keypoints is not None:
            return keypoints.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          keypoints (torchTensor or none): point coordinates and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(keypoints, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        sparse_masks = torch.empty((bs, 0), device=self._get_device())
        if keypoints is not None:
            coords = keypoints[:, :, :2]
            labels = keypoints[:, :, -1]
            point_embeddings, point_mask = self._embed_keypoints(
                coords, labels
            )
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            sparse_masks = torch.cat([sparse_masks, point_mask], dim=1)

        return sparse_embeddings, sparse_masks

    def get_mask_embeddings(
        self,
        masks: Optional[torch.Tensor] = None,
        bs: int = 1,
        size: Tuple[int, int] = (16, 16),
    ) -> torch.Tensor:
        """Embeds mask inputs."""
        no_mask_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, size[0], size[1]
        )
        if masks is not None:
            mask_embeddings = self.mask_downscaling(masks)
        else:
            mask_embeddings = no_mask_embeddings
        return mask_embeddings, no_mask_embeddings

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""

        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords

        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))

```

* Файл: sam_3d_body/models/heads/camera_head.py

```python

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sam_3d_body.models.modules.geometry_utils import perspective_projection

from ..modules import to_2tuple
from ..modules.transformer import FFN

class PerspectiveHead(nn.Module):
    """
    Predict camera translation (s, tx, ty) and perform full-perspective
    2D reprojection (CLIFF/CameraHMR setup).
    """

    def __init__(
        self,
        input_dim: int,
        img_size: Tuple[int, int],
        mlp_depth: int = 1,
        drop_ratio: float = 0.0,
        mlp_channel_div_factor: int = 8,
        default_scale_factor: float = 1,
    ):
        super().__init__()


        self.img_size = to_2tuple(img_size)
        self.ncam = 3
        self.default_scale_factor = default_scale_factor

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.ncam,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.ncam]
        """
        pred_cam = self.proj(x)
        if init_estimate is not None:
            pred_cam = pred_cam + init_estimate

        return pred_cam

    def perspective_projection(
        self,
        points_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_size: torch.Tensor,
        img_size: torch.Tensor,
        cam_int: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Args:
            bbox_center / img_size: shape [N, 2], in original image space (w, h)
            bbox_size: shape [N,], in original image space
            cam_int: shape [N, 3, 3]
        """
        batch_size = points_3d.shape[0]
        pred_cam = pred_cam.clone()
        pred_cam[..., [0, 2]] *= -1




        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bs = bbox_size * s * self.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
            cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)


        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)



        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d.reshape(batch_size, -1, 2),
            "pred_cam_t": pred_cam_t,
            "focal_length": focal_length,
            "pred_keypoints_2d_depth": j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
        }

```

* Файл: sam_3d_body/models/heads/__init__.py

```python

from ..modules import to_2tuple
from .camera_head import PerspectiveHead
from .mhr_head import MHRHead

def build_head(cfg, head_type="mhr", enable_hand_model=False, default_scale_factor=1.0):
    if head_type == "mhr":
        return MHRHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.MHR_HEAD.get("MLP_DEPTH", 1),
            mhr_model_path=cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH,
            mlp_channel_div_factor=cfg.MODEL.MHR_HEAD.get("MLP_CHANNEL_DIV_FACTOR", 1),
            enable_hand_model=enable_hand_model,
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            default_scale_factor=default_scale_factor,
        )
    else:
        raise ValueError("Invalid head type: ", head_type)

```

* Файл: sam_3d_body/models/heads/mhr_head.py

```python

import os
import warnings
from typing import Optional

import roma
import torch
import torch.nn as nn

from ..modules import rot6d_to_rotmat
from ..modules.mhr_utils import (
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    compact_model_params_to_cont_body,
    mhr_param_hand_mask,
)

from ..modules.transformer import FFN

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR

        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")

class MHRHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        mhr_model_path: str = "",
        extra_joint_regressor: str = "",
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
        enable_hand_model=False,
    ):
        super().__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model

        self.body_cont_dim = 260
        self.npose = (
            6
            + self.body_cont_dim
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)


        self.model_data_dir = mhr_model_path
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps


        self.joint_rotation = nn.Parameter(torch.zeros(127, 3, 3), requires_grad=False)
        self.scale_mean = nn.Parameter(torch.zeros(68), requires_grad=False)
        self.scale_comps = nn.Parameter(torch.zeros(28, 68), requires_grad=False)
        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
        self.hand_pose_mean = nn.Parameter(torch.zeros(54), requires_grad=False)
        self.hand_pose_comps = nn.Parameter(torch.eye(54), requires_grad=False)
        self.hand_joint_idxs_left = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.hand_joint_idxs_right = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.keypoint_mapping = nn.Parameter(
            torch.zeros(308, 18439 + 127), requires_grad=False
        )

        self.right_wrist_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.root_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.local_to_world_wrist = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
        self.nonhand_param_idxs = nn.Parameter(
            torch.zeros(145).long(), requires_grad=False
        )


        if MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                lod=1,
            )
        else:
            self.mhr = torch.jit.load(
                mhr_model_path,
                map_location=("cuda" if torch.cuda.is_available() else "cpu"),
            )

        for param in self.mhr.parameters():
            param.requires_grad = False

    def get_zero_pose_init(self, factor=1.0):


        weights = torch.zeros(1, self.npose)
        weights[:, : 6 + self.body_cont_dim] = torch.cat(
            [
                torch.FloatTensor([1, 0, 0, 0, 1, 0]),
                compact_model_params_to_cont_body(torch.zeros(1, 133)).squeeze()
                * factor,
            ],
            dim=0,
        )
        return weights

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136



        left_hand_params, right_hand_params = torch.split(
            hand_pose_params,
            [self.num_hand_pose_comps, self.num_hand_pose_comps],
            dim=1,
        )


        left_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", left_hand_params, self.hand_pose_comps)
        )
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", right_hand_params, self.hand_pose_comps)
        )


        full_pose_params[:, self.hand_joint_idxs_left] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_hand_params_model_params

        return full_pose_params

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        do_pcblend=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
        scale_offsets=None,
        vertex_offsets=None,
    ):
        if self.enable_hand_model:

            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = roma.rotmat_to_euler(
                "xyz",
                roma.euler_to_rotmat("xyz", global_rot_ori) @ self.local_to_world_wrist,
            )
            global_trans = (
                -(
                    roma.euler_to_rotmat("xyz", global_rot)
                    @ (self.right_wrist_coords - self.root_coords)
                    + self.root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]


        * Add singleton batches in case...
        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        * Convert scale...
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets


        * 10 here is because it's more stable to optimize global translation in meters.
        full_pose_params = torch.cat(
            [global_trans * 10, global_rot, body_pose_params], dim=1
        )
        * Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )
        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:

            model_params[:, self.nonhand_param_idxs] = 0

        curr_skinned_verts, curr_skel_state = self.mhr(
            shape_params, model_params, expr_params
        )
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100
        curr_joint_rots = roma.unitquat_to_rotmat(curr_joint_quats)


        to_return = [curr_skinned_verts]
        if return_keypoints:

            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )
            model_keypoints_pred = (
                (
                    self.keypoint_mapping
                    @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
                )
                .reshape(-1, model_vert_joints.shape[0], 3)
                .permute(1, 0, 2)
            )

            if self.enable_hand_model:

                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        do_pcblend=True,
        slim_keypoints=False,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
        """
        batch_size = x.shape[0]
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate



        * First, get globals

        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
        global_trans = torch.zeros_like(global_rot_euler)

        * Next, get body pose.

        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim

        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)

        pred_pose_euler[:, mhr_param_hand_mask] = 0

        pred_pose_euler[:, -3:] = 0

        * Get remaining parameters
        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps


        output = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=do_pcblend,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )


        verts, j3d, jcoords, mhr_model_params, joint_global_rots = output


        if verts is not None:
            verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1


        output = {
            "pred_pose_raw": torch.cat(
                [global_rot_6d, pred_pose_cont], dim=1
            ),
            "pred_pose_rotmat": None,
            "global_rot": global_rot_euler,
            "body_pose": pred_pose_euler,
            "shape": pred_shape,
            "scale": pred_scale,
            "hand": pred_hand,
            "face": pred_face,
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": (
                verts.reshape(batch_size, -1, 3) if verts is not None else None
            ),
            "pred_joint_coords": (
                jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
            ),
            "faces": self.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }

        return output

```

* Файл: sam_3d_body/models/__init__.py

```python

```

* Файл: sam_3d_body/models/meta_arch/base_lightning_module.py

```python

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

class BaseLightningModule(pl.LightningModule):
    def _log_metric(self, name, value, step=None):
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                if step is not None:
                    logger.experiment.log({name: value, "step": step})
                else:
                    logger.experiment.log({name: value})
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_scalar(name, value, step)
            else:
                raise ValueError(f"Unsupported logger: {logger}")

    def _log_image(self, name, img_tensor, dataformats="CHW", step_count=None):
        """Log image tensor to both W&B and TensorBoard."""
        step = step_count if step_count is not None else self.global_step
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                img = img_tensor
                if dataformats.upper() == "CHW":

                    img = img_tensor.permute(1, 2, 0).cpu().numpy()
                logger.experiment.log({name: wandb.Image(img), "step": step})
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    name, img_tensor, step, dataformats=dataformats
                )
            else:
                raise ValueError(f"Unsupported logger: {logger}")

    def _log_hist(self, name, array, step_count=None):
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                value = wandb.Histogram(
                    np_histogram=(array, np.arange(array.shape[0] + 1)),
                )
                logger.experiment.log({name: value, "step": step_count})

```

* Файл: sam_3d_body/models/meta_arch/base_model.py

```python

"""Define an abstract base model for consistent format input / processing / output."""

from abc import abstractmethod
from functools import partial
from typing import Dict, Optional

import torch
from yacs.config import CfgNode

from ..optim.fp16_utils import convert_module_to_f16, convert_to_fp16_safe

from .base_lightning_module import BaseLightningModule

class BaseModel(BaseLightningModule):
    def __init__(self, cfg: Optional[CfgNode], **kwargs):
        super().__init__()


        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self._initialze_model(**kwargs)


        self._max_num_person = None
        self._person_valid = None

    @abstractmethod
    def _initialze_model(self, **kwargs) -> None:
        pass

    def data_preprocess(
        self,
        inputs: torch.Tensor,
        crop_width: bool = False,
        is_full: bool = False,
        crop_hand: int = 0,
    ) -> torch.Tensor:
        image_mean = self.image_mean if not is_full else self.full_image_mean
        image_std = self.image_std if not is_full else self.full_image_std

        if inputs.max() > 1 and image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and image_mean.max() > 1:
            inputs = inputs * 255.0
        batch_inputs = (inputs - image_mean) / image_std

        if crop_width:
            if crop_hand > 0:
                batch_inputs = batch_inputs[:, :, :, crop_hand:-crop_hand]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
            ]:

                batch_inputs = batch_inputs[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                batch_inputs = batch_inputs[:, :, :, 64:-64]
            else:
                raise Exception

        return batch_inputs

    def _initialize_batch(self, batch: Dict) -> None:


        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
            self._person_valid = self._flatten_person(batch["person_valid"]) > 0
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0
            self._person_valid = None

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._max_num_person:

            shape = x.shape
            x = x.view(self._batch_size * self._max_num_person, *shape[2:])
        return x

    def _unflatten_person(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self._max_num_person:
            x = x.view(self._batch_size, self._max_num_person, *shape[1:])
        return x

    def _get_valid(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None, "No max_num_person initialized"

        if self._person_valid is not None:
            x = x[self._person_valid]
        return x

    def _full_to_crop(
        self, batch: Dict, pred_keypoints_2d: torch.Tensor
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        affine_trans = self._flatten_person(batch["affine_trans"]).to(
            pred_keypoints_2d_cropped
        )
        img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def _cam_full_to_crop(
        self, batch: Dict, pred_cam_t: torch.Tensor, focal_length: torch.Tensor = None
    ) -> torch.Tensor:
        """Revert the camera translation from full to crop image space"""
        num_person = batch["img"].shape[1]
        cam_int = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, num_person, -1, -1).contiguous()
        )
        bbox_center = self._flatten_person(batch["bbox_center"])
        bbox_size = self._flatten_person(batch["bbox_scale"])[:, 0]
        img_size = self._flatten_person(batch["ori_img_size"])
        input_size = self._flatten_person(batch["img_size"])[:, 0]

        tx, ty, tz = pred_cam_t[:, 0], pred_cam_t[:, 1], pred_cam_t[:, 2]
        if focal_length is None:
            focal_length = cam_int[:, 0, 0]
        bs = 2 * focal_length / (tz + 1e-8)

        cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
        cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        crop_cam_t = torch.stack(
            [tx - cx, ty - cy, tz * bbox_size / input_size], dim=-1
        )
        return crop_cam_t

    def convert_to_fp16(self) -> torch.dtype:
        """
        Convert the torso of the model to float16.
        """
        fp16_type = (
            torch.float16
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16"
            else torch.bfloat16
        )

        if hasattr(self, "backbone"):
            self._set_fp16(self.backbone, fp16_type)
        if hasattr(self, "full_encoder"):
            self._set_fp16(self.full_encoder, fp16_type)

        if hasattr(self.backbone, "lhand_pos_embed"):
            self.backbone.lhand_pos_embed.data = self.backbone.lhand_pos_embed.data.to(
                fp16_type
            )

        if hasattr(self.backbone, "rhand_pos_embed"):
            self.backbone.rhand_pos_embed.data = self.backbone.rhand_pos_embed.data.to(
                fp16_type
            )

        return fp16_type

    def _set_fp16(self, module, fp16_type):
        if hasattr(module, "pos_embed"):
            module.apply(partial(convert_module_to_f16, dtype=fp16_type))
            module.pos_embed.data = module.pos_embed.data.to(fp16_type)
        elif hasattr(module.encoder, "rope_embed"):

            module.encoder.apply(partial(convert_to_fp16_safe, dtype=fp16_type))
            module.encoder.rope_embed = module.encoder.rope_embed.to(fp16_type)
        else:

            module.encoder.pos_embed.data = module.encoder.pos_embed.data.to(fp16_type)

```

* Файл: sam_3d_body/models/meta_arch/__init__.py

```python

from .sam3d_body import SAM3DBody

```

* Файл: sam_3d_body/models/meta_arch/sam3d_body.py

```python

from typing import Any, Dict, Optional, Tuple

import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.models.decoders.prompt_encoder import PositionEmbeddingRandom
from sam_3d_body.models.modules.mhr_utils import (
    fix_wrist_euler,
    rotation_angle_difference,
)
from sam_3d_body.utils import recursive_to
from sam_3d_body.utils.logging import get_pylogger

from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
from ..modules.camera_embed import CameraEncoder
from ..modules.transformer import FFN, MLP

from .base_model import BaseModel

logger = get_pylogger(__name__)

PROMPT_KEYPOINTS = {
    "mhr70": {
        i: i for i in range(70)
    },
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
KEY_RIGHT_HAND = list(range(21, 42))

class SAM3DBody(BaseModel):
    pelvis_idx = [9, 10]

    def _initialze_model(self):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )


        self.backbone = create_backbone(self.cfg.MODEL.BACKBONE.TYPE, self.cfg)


        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        self.head_pose.hand_pose_comps_ori = nn.Parameter(
            self.head_pose.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        )



        self.init_pose = nn.Embedding(1, self.head_pose.npose)


        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE)
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"


        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_mhr = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.init_to_token_mhr_hand = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr_hand = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )


        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
            )

            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False


        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )

        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)


        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(
                2, self.cfg.MODEL.DECODER.DIM
            )

            self.hand_cls_embed = nn.Linear(self.cfg.MODEL.DECODER.DIM, 2)
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3
            )

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_feat_linear_hand = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )


        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )


        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":

            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )
            condition_info[:, 2] = condition_info[:, 2] / focal_length
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]


        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr(init_input).view(
            batch_size, 1, -1
        )

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:

                prev_estimate = init_estimate

            prev_embeddings = self.prev_to_token_mhr(prev_estimate).view(
                batch_size, 1, -1
            )

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:

                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:

                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )

            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])



            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )


            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):

                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )

                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)

            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )

            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):

                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )

                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )


        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):

            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)


            pose_output = self.head_pose(pose_token, prev_pose)

            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam

            pose_output = self.camera_project(pose_output, batch)


            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.body_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn


        kp3d_token_update_fn = self.keypoint3d_token_update_fn


        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def forward_decoder_hand(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]


        if init_estimate is None:
            init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(
                dim=1
            )
            if hasattr(self, "init_camera_hand"):
                init_camera = self.init_camera_hand.weight.expand(
                    batch_size, -1
                ).unsqueeze(dim=1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera_hand")
                else torch.cat([init_pose, init_camera], dim=-1)
            )

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr_hand(init_input).view(
            batch_size, 1, -1
        )
        num_pose_token = token_embeddings.shape[1]

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:

                prev_estimate = init_estimate

            prev_embeddings = self.prev_to_token_mhr_hand(prev_estimate).view(
                batch_size, 1, -1
            )

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:

                image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:

                image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
            else:
                image_augment = self.hand_pe_layer(
                    image_embeddings.shape[-2:]
                ).unsqueeze(0)

            image_embeddings = self.ray_cond_emb_hand(
                image_embeddings, batch["ray_cond_hand"]
            )



            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )


            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):

                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )

                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)

            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding_hand.weight[None, :, :].repeat(
                        batch_size, 1, 1
                    ),
                ],
                dim=1,
            )

            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):

                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_hand.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )

                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )


        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):

            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)


            pose_output = self.head_pose_hand(pose_token, prev_pose)


            if hasattr(self, "head_camera_hand"):
                pred_cam = self.head_camera_hand(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam

            pose_output = self.camera_project_hand(pose_output, batch)


            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.hand_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn_hand


        kp3d_token_update_fn = self.keypoint3d_token_update_fn_hand


        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder_hand(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
        ]:

            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:

            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _one_prompt_iter(self, batch, output, prev_prompt, full_output):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]

        if "mhr" in output and output["mhr"] is not None:
            pose_output = output["mhr"]

            prev_estimate = torch.cat(
                [
                    pose_output["pred_pose_raw"].detach(),
                    pose_output["shape"].detach(),
                    pose_output["scale"].detach(),
                    pose_output["hand"].detach(),
                    pose_output["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                prev_estimate = torch.cat(
                    [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                    dim=-1,
                )
            prev_shape = prev_estimate.shape[1:]

            pred_keypoints_2d = output["mhr"]["pred_keypoints_2d"].detach().clone()
            kpt_shape = pred_keypoints_2d.shape[1:]

        if "mhr_hand" in output and output["mhr_hand"] is not None:
            pose_output_hand = output["mhr_hand"]

            prev_estimate_hand = torch.cat(
                [
                    pose_output_hand["pred_pose_raw"].detach(),
                    pose_output_hand["shape"].detach(),
                    pose_output_hand["scale"].detach(),
                    pose_output_hand["hand"].detach(),
                    pose_output_hand["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera_hand"):
                prev_estimate_hand = torch.cat(
                    [
                        prev_estimate_hand,
                        pose_output_hand["pred_cam"].detach().unsqueeze(1),
                    ],
                    dim=-1,
                )
            prev_shape = prev_estimate_hand.shape[1:]

            pred_keypoints_2d_hand = (
                output["mhr_hand"]["pred_keypoints_2d"].detach().clone()
            )
            kpt_shape = pred_keypoints_2d_hand.shape[1:]

        all_prev_estimate = torch.zeros(
            (image_embeddings.shape[0], *prev_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_prev_estimate[self.body_batch_idx] = prev_estimate
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_prev_estimate[self.hand_batch_idx] = prev_estimate_hand


        all_pred_keypoints_2d = torch.zeros(
            (image_embeddings.shape[0], *kpt_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_pred_keypoints_2d[self.body_batch_idx] = pred_keypoints_2d
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_pred_keypoints_2d[self.hand_batch_idx] = pred_keypoints_2d_hand

        keypoint_prompt = self._get_keypoint_prompt(batch, all_pred_keypoints_2d)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt

        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=cur_keypoint_prompt[self.body_batch_idx],
                prev_estimate=all_prev_estimate[self.body_batch_idx],
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output = pose_output[-1]


        output.update(
            {
                "mhr": pose_output,
                "mhr_hand": pose_output_hand,
            }
        )

        return output, keypoint_prompt

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.body_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def camera_project_hand(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera_hand"):
            head_camera = self.head_camera_hand
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.hand_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.hand_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .cuda()
        )
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )


        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        batch_size, num_person = batch["img"].shape[:2]


        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                ]
            ),
        )


        ray_cond = self.get_ray_condition(batch)
        ray_cond = self._flatten_person(ray_cond)
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
        ]:
            ray_cond = ray_cond[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            ray_cond = ray_cond[:, :, :, 64:-64]

        if len(self.body_batch_idx):
            batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
        if len(self.hand_batch_idx):
            batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
        ray_cond = None

        image_embeddings = self.backbone(
            x.type(self.backbone_dtype), extra_embed=ray_cond
        )

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)


        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:

            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError


        condition_info = self._get_decoder_condition(batch)


        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2


        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]
        if len(self.hand_batch_idx):
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
            )
            pose_output_hand = pose_output_hand[-1]

        output = {

            "mhr": pose_output,
            "mhr_hand": pose_output_hand,
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            if len(self.body_batch_idx):
                output_hand_box_tokens = tokens_output
                hand_coords = self.bbox_embed(
                    output_hand_box_tokens
                ).sigmoid()
                hand_logits = self.hand_cls_embed(output_hand_box_tokens)

                output["mhr"]["hand_box"] = hand_coords
                output["mhr"]["hand_logits"] = hand_logits

            if len(self.hand_batch_idx):
                output_hand_box_tokens_hand_batch = tokens_output_hand

                hand_coords_hand_batch = self.bbox_embed(
                    output_hand_box_tokens_hand_batch
                ).sigmoid()
                hand_logits_hand_batch = self.hand_cls_embed(
                    output_hand_box_tokens_hand_batch
                )

                output["mhr_hand"]["hand_box"] = hand_coords_hand_batch
                output["mhr_hand"]["hand_logits"] = hand_logits_hand_batch

        return output

    def forward_step(
        self, batch: Dict, decoder_type: str = "body"
    ) -> Tuple[Dict, Dict]:
        batch_size, num_person = batch["img"].shape[:2]

        if decoder_type == "body":
            self.hand_batch_idx = []
            self.body_batch_idx = list(range(batch_size * num_person))
        elif decoder_type == "hand":
            self.hand_batch_idx = list(range(batch_size * num_person))
            self.body_batch_idx = []
        else:
            ValueError("Invalid decoder type: ", decoder_type)


        pose_output = self.forward_pose_branch(batch)

        return pose_output

    def run_inference(
        self,
        img,
        batch: Dict,
        inference_type: str = "full",
        transform_hand: Any = None,
        thresh_wrist_angle=1.4,
    ):
        """
        Run 3DB inference (optionally with hand detector).

        inference_type:
            - full: full-body inference with both body and hand decoders
            - body: inference with body decoder only (still full-body output)
            - hand: inference with hand decoder only (only hand output)
        """

        height, width = img.shape[:2]
        cam_int = batch["cam_int"].clone()

        if inference_type == "body":
            pose_output = self.forward_step(batch, decoder_type="body")
            return pose_output
        elif inference_type == "hand":
            pose_output = self.forward_step(batch, decoder_type="hand")
            return pose_output
        elif not inference_type == "full":
            ValueError("Invalid inference type: ", inference_type)


        pose_output = self.forward_step(batch, decoder_type="body")
        left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
                1, (2, 3)
            ),
        )


        * Left... Flip image & box
        flipped_img = img[:, ::-1]
        tmp = left_xyxy.copy()
        left_xyxy[:, 0] = width - tmp[:, 2] - 1
        left_xyxy[:, 2] = width - tmp[:, 0] - 1

        batch_lhand = prepare_batch(
            flipped_img, transform_hand, left_xyxy, cam_int=cam_int.clone()
        )
        batch_lhand = recursive_to(batch_lhand, "cuda")
        lhand_output = self.forward_step(batch_lhand, decoder_type="hand")


        * Flip scale

        scale_r_hands_mean = self.head_pose.scale_mean[8].item()
        scale_l_hands_mean = self.head_pose.scale_mean[9].item()
        scale_r_hands_std = self.head_pose.scale_comps[8, 8].item()
        scale_l_hands_std = self.head_pose.scale_comps[9, 9].item()

        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (
                scale_r_hands_mean
                + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
            )
            - scale_l_hands_mean
        ) / scale_l_hands_std
        * Get the right hand global rotation, flip it, put it in as left.
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
            "joint_global_rots"
        ][:, 42].clone()
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1

        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][
            :, 54:
        ]

        batch_lhand["bbox_center"][:, :, 0] = (
            width - batch_lhand["bbox_center"][:, :, 0] - 1
        )

        * Right...
        batch_rhand = prepare_batch(
            img, transform_hand, right_xyxy, cam_int=cam_int.clone()
        )
        batch_rhand = recursive_to(batch_rhand, "cuda")
        rhand_output = self.forward_step(batch_rhand, decoder_type="hand")


        * CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
        joint_rotations = pose_output["mhr"]["joint_global_rots"]

        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]

        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )

        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )

        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )
        angle_difference_valid_mask = angle_difference < thresh_wrist_angle

        * CRITERIA 2: hand box size
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack(
            [
                (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
                (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
            ],
            dim=1,
        )

        * CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
        hand_kps2d_thresh = 0.5
        hand_kps2d_valid_mask = torch.stack(
            [
                lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
                rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
            ],
            dim=1,
        )

        * CRITERIA 4: 2D wrist distance.
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1
        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_wrist_idx]
        ].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack(
            [
                left_kps_dist < hand_wrist_kps2d_thresh,
                right_kps_dist < hand_wrist_kps2d_thresh,
            ],
            dim=1,
        )
        * Left-right
        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )




        batch_size, num_person = batch["img"].shape[:2]
        self.hand_batch_idx = []
        self.body_batch_idx = list(range(batch_size * num_person))

        * Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1


        right_kps_crop = self._full_to_crop(batch, right_kps_full)
        left_kps_crop = self._full_to_crop(batch, left_kps_full)


        kps_right_elbow_idx = 8
        kps_left_elbow_idx = 7
        right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_elbow_idx]
        ].clone()
        left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_elbow_idx]
        ].clone()


        right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
        left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)


        keypoint_prompt = torch.cat(
            [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
            dim=1,
        )
        keypoint_prompt = torch.cat(
            [keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1
        )
        keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
        keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
        keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
        keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

        if keypoint_prompt.shape[0] > 1:

            invalid_prompt = (
                (keypoint_prompt[..., 0] < -0.5)
                | (keypoint_prompt[..., 0] > 0.5)
                | (keypoint_prompt[..., 1] < -0.5)
                | (keypoint_prompt[..., 1] > 0.5)
                | (~hand_valid_mask[..., [1, 0, 1, 0]])
            ).unsqueeze(-1)
            dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
            dummy_prompt[:, :, -1] = -2
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )
            keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
        else:

            valid_keypoint = (
                torch.all(
                    (keypoint_prompt[:, :, :2] > -0.5)
                    & (keypoint_prompt[:, :, :2] < 0.5),
                    dim=2,
                )
                & hand_valid_mask[..., [1, 0, 1, 0]]
            ).squeeze()
            keypoint_prompt = keypoint_prompt[:, valid_keypoint]
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )

        if keypoint_prompt.numel() != 0:
            pose_output, _ = self.run_keypoint_prompt(
                batch, pose_output, keypoint_prompt
            )




        left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
        right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
        updated_hand_pose = torch.cat(
            [left_hand_pose_params, right_hand_pose_params], dim=1
        )


        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2


        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2




        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]


        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]


        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )


        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )


        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        wrist_xzy = fix_wrist_euler(
            roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat)
        )


        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )
        valid_angle = angle_difference < thresh_wrist_angle
        valid_angle = valid_angle & hand_valid_mask
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][
            :, [41, 43, 42, 31, 33, 32]
        ].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
            updated_body_pose.flatten(1, 2)
        )

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
            dim=1,
        )
        updated_hand_scale = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_hand_scale = torch.where(
            valid_angle.squeeze(-1), updated_hand_scale, hand_scale
        )
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]


        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )




        with torch.no_grad():
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
                self.head_pose.mhr_forward(
                    global_trans=pose_output["mhr"]["global_rot"] * 0,
                    global_rot=pose_output["mhr"]["global_rot"],
                    body_pose_params=pose_output["mhr"]["body_pose"],
                    hand_pose_params=pose_output["mhr"]["hand"],
                    scale_params=pose_output["mhr"]["scale"],
                    shape_params=pose_output["mhr"]["shape"],
                    expr_params=pose_output["mhr"]["face"],
                    return_keypoints=True,
                    return_joint_coords=True,
                    return_model_params=True,
                    return_joint_rotations=True,
                )
            )
            j3d = j3d[:, :70]
            verts[..., [1, 2]] *= -1
            j3d[..., [1, 2]] *= -1
            jcoords[..., [1, 2]] *= -1
            pose_output["mhr"]["pred_keypoints_3d"] = j3d
            pose_output["mhr"]["pred_vertices"] = verts
            pose_output["mhr"]["pred_joint_coords"] = jcoords
            pose_output["mhr"]["pred_pose_raw"][...] = (
                0
            )
            pose_output["mhr"]["mhr_model_params"] = mhr_model_params



        pred_keypoints_3d_proj = (
            pose_output["mhr"]["pred_keypoints_3d"]
            + pose_output["mhr"]["pred_cam_t"][:, None, :]
        )
        pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
            :, None, None
        ]
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
                None, None, :
            ]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

        return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]

        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )

        tokens_output, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Get hand bbox from the hand detector"""
        pred_left_hand_box = (
            pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )
        pred_right_hand_box = (
            pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )


        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = (
            pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = (
            pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )


        batch["left_scale"] = (
            batch["left_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["right_scale"] = (
            batch["right_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["left_center"] = (
            batch["left_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        batch["right_center"] = (
            batch["right_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        left_xyxy = np.concatenate(
            [
                (
                    batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (
                    batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )

        return left_xyxy, right_xyxy

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):

        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx


        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding.weight.shape[0]


        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]


        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5


        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )


        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )




        * Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:


            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )


        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )

        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )

        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):

        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]


        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()


        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )


        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]


        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):

        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx


        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]


        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
        ]


        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5


        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )


        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )




        * Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:


            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )


        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )

        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )

        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):

        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]


        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()


        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )


        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]


        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

```

* Файл: sam_3d_body/models/modules/camera_embed.py

```python

import einops
import numpy as np
import torch
import torch.nn.functional as F

from sam_3d_body.models.modules.transformer import LayerNorm2d
from torch import nn

class CameraEncoder(nn.Module):
    def __init__(self, embed_dim, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)

        self.conv = nn.Conv2d(embed_dim + 99, embed_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, img_embeddings, rays):
        B, D, _h, _w = img_embeddings.shape

        with torch.no_grad():
            scale = 1 / self.patch_size
            rays = F.interpolate(
                rays,
                scale_factor=(scale, scale),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            rays = rays.permute(0, 2, 3, 1).contiguous()
            rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
            rays_embeddings = self.camera(
                pos=rays.reshape(B, -1, 3)
            )
            rays_embeddings = einops.rearrange(
                rays_embeddings, "b (h w) c -> b c h w", h=_h, w=_w
            ).contiguous()

        z = torch.concat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z

class FourierPositionEncoding(nn.Module):
    def __init__(self, n, num_bands, max_resolution):
        """
        Module that generate Fourier encoding - no learning involved
        """
        super().__init__()

        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    @property
    def channels(self):
        """
        Return the output dimension
        """
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        encoding_size *= 2
        encoding_size += num_dims

        return encoding_size

    def forward(self, pos):
        """
        Forward pass that take rays as input and generate Fourier positional encodings
        """
        fourier_pos_enc = _generate_fourier_features(
            pos, num_bands=self.num_bands, max_resolution=self.max_resolution
        )
        return fourier_pos_enc

def _generate_fourier_features(pos, num_bands, max_resolution):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device


    min_freq = 1.0
    freq_bands = torch.stack(
        [
            torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
            for res in max_resolution
        ],
        dim=0,
    )


    per_pos_features = torch.stack(
        [pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0
    )
    per_pos_features = per_pos_features.reshape(b, n, -1)


    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
        dim=-1,
    )


    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features

```

* Файл: sam_3d_body/models/modules/drop_path.py

```python

import torch
import torch.nn as nn

def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    if not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

```

* Файл: sam_3d_body/models/modules/geometry_utils.py

```python

from typing import Optional

import cv2

import numpy as np
import torch
from torch.nn import functional as F

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):

    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    if type(focal_length) is float:
        focal_length = torch.ones_like(cam_bbox[:, 0]) * focal_length
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).

    Alternatives:
        import roma
        y = roma.rotvec_to_rotmat(x)
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return _quat_to_rotmat(quat)

def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).

    Alternatives:
        import roma
        x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
        y = roma.special_gramschmidt(x)
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def rotmat_to_rot6d(x: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        x: batch of rotation matrices of size (B, 3, 3)

    Returns:
        6D rotation representation, of size (B, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = x.size()[:-2]
    return x[..., :2, :].clone().reshape(batch_dim + (6,))

def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """

    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )

    per_rdg, _ = cv2.Rodrigues(aa)

    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)

def transform_points(
    points: torch.Tensor,
    translation: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transform a set of 3D points given translation and rotation.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) containing the transformed points.
    """
    if rotation is not None:
        points = torch.einsum("bij,bkj->bki", rotation, points)

    if translation is not None:
        points = points + translation.unsqueeze(1)

    return points

def get_intrinsic_matrix(
    focal_length: torch.Tensor, principle: torch.Tensor
) -> torch.Tensor:
    """
    Populate intrinsic camera matrix K given focal length and principle point.
    Args:
        focal_length: Tensor of shape (2,)
        principle: Tensor of shape (2,)
    Returns:
        Tensor of shape (3, 3)
    """
    if isinstance(focal_length, float):
        fl_x = fl_y = focal_length
    elif len(focal_length) == 1:
        fl_x = fl_y = focal_length[0]
    else:
        fl_x, fl_y = focal_length[0], focal_length[1]
    K = torch.eye(3)
    K[0, 0] = fl_x
    K[1, 1] = fl_y
    K[0, -1] = principle[0]
    K[1, -1] = principle[1]

    return K

def perspective_projection(x, K):
    """
    Computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """

    y = x / x[:, :, -1].unsqueeze(-1)


    y = torch.einsum("bij,bkj->bki", K, y)

    return y[:, :, :2]

def inverse_perspective_projection(points, K, distance):
    """
    Computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        K (bs,3,3): camera intrinsics params
        distance (bs, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """

    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(K), points)


    if distance == None:
        return points
    points = points * distance
    return points

def get_cam_intrinsics(img_size, fov=55, p_x=None, p_y=None):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = np.eye(3)

    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal


    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    return K

def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    return focal

def focal_length_normalization(x, f, fovn=60, img_size=448):
    """
    Section 3.1 of https://arxiv.org/pdf/1904.02028.pdf
    E = (fn/f) * E' where E is 1/d
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    y = x * (fn / f)
    return y

def undo_focal_length_normalization(y, f, fovn=60, img_size=448):
    """
    Undo focal_length_normalization()
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    x = y * (f / fn)
    return x

EPS_LOG = 1e-10

def log_depth(x, eps=EPS_LOG):
    """
    Move depth to log space
    """
    return torch.log(x + eps)

def undo_log_depth(y, eps=EPS_LOG):
    """
    Undo log_depth()
    """
    return torch.exp(y) - eps

```

* Файл: sam_3d_body/models/modules/__init__.py

```python

from .geometry_utils import (
    aa_to_rotmat,
    cam_crop_to_full,
    focal_length_normalization,
    get_focalLength_from_fieldOfView,
    get_intrinsic_matrix,
    inverse_perspective_projection,
    log_depth,
    perspective_projection,
    rot6d_to_rotmat,
    transform_points,
    undo_focal_length_normalization,
    undo_log_depth,
)

from .misc import to_2tuple, to_3tuple, to_4tuple, to_ntuple

```

* Файл: sam_3d_body/models/modules/layer_scale.py

```python

from typing import Union

import torch
import torch.nn as nn

class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
        data_format: str = "channels_last",
    ):
        super().__init__()
        assert data_format in (
            "channels_last",
            "channels_first",
        ), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == "channels_first":
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight

```

* Файл: sam_3d_body/models/modules/mhr_utils.py

```python

import cv2

import torch
import torch.nn.functional as F

def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """

    R_rel = torch.matmul(A, B.transpose(-2, -1))

    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

    cos_theta = (trace - 1) / 2

    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)

    angle = torch.acos(cos_theta_clamped)
    return angle

def fix_wrist_euler(
    wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5), limits_y=(-1.2, 1.5)
):
    """
    wrist_xzy: B x 2 x 3 (X, Z, Y angles)
    Returns: Fixed angles within joint limits
    """
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]

    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))


    def calc_violation(val, limits):
        below = torch.clamp(limits[0] - val, min=0.0)
        above = torch.clamp(val - limits[1], min=0.0)
        return below**2 + above**2

    violation_orig = (
        calc_violation(x, limits_x)
        + calc_violation(z, limits_z)
        + calc_violation(y, limits_y)
    )

    violation_alt = (
        calc_violation(x_alt, limits_x)
        + calc_violation(z_alt, limits_z)
        + calc_violation(y_alt, limits_y)
    )


    use_alt = violation_alt < violation_orig


    wrist_xzy_alt = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    result = torch.where(use_alt.unsqueeze(-1), wrist_xzy_alt, wrist_xzy)

    return result

def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result

def batchXYZfrom6D(poses):


    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)



    sy = torch.sqrt(
        matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0]
    )
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0

    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x * (1 - singular) + xs * singular
    out_euler[..., 1] = y * (1 - singular) + ys * singular
    out_euler[..., 2] = z * (1 - singular) + zs * singular

    return out_euler

def resize_image(image_array, scale_factor, interpolation=cv2.INTER_LINEAR):
    new_height = int(image_array.shape[0] // scale_factor)
    new_width = int(image_array.shape[1] // scale_factor)
    resized_image = cv2.resize(
        image_array, (new_width, new_height), interpolation=interpolation
    )

    return resized_image

def compact_cont_to_model_params_hand(hand_cont):

    assert hand_cont.shape[-1] == 54
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27

    mask_cont_threedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )

    mask_cont_onedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )

    mask_model_params_threedofs = torch.cat(
        [torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )

    mask_model_params_onedofs = torch.cat(
        [torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )


    * First for 3DoFs
    hand_cont_threedofs = hand_cont[..., mask_cont_threedofs].unflatten(-1, (-1, 6))
    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    * Next for 1DoFs
    hand_cont_onedofs = hand_cont[..., mask_cont_onedofs].unflatten(
        -1, (-1, 2)
    )
    hand_model_params_onedofs = torch.atan2(
        hand_cont_onedofs[..., -2], hand_cont_onedofs[..., -1]
    )


    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27).to(hand_cont)
    hand_model_params[..., mask_model_params_threedofs] = hand_model_params_threedofs
    hand_model_params[..., mask_model_params_onedofs] = hand_model_params_onedofs

    return hand_model_params

def compact_model_params_to_cont_hand(hand_model_params):

    assert hand_model_params.shape[-1] == 27
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27

    mask_cont_threedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )

    mask_cont_onedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )

    mask_model_params_threedofs = torch.cat(
        [torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )

    mask_model_params_onedofs = torch.cat(
        [torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )


    * First for 3DoFs
    hand_model_params_threedofs = hand_model_params[
        ..., mask_model_params_threedofs
    ].unflatten(-1, (-1, 3))
    hand_cont_threedofs = batch6DFromXYZ(hand_model_params_threedofs).flatten(-2, -1)
    * Next for 1DoFs
    hand_model_params_onedofs = hand_model_params[..., mask_model_params_onedofs]
    hand_cont_onedofs = torch.stack(
        [hand_model_params_onedofs.sin(), hand_model_params_onedofs.cos()], dim=-1
    ).flatten(-2, -1)


    hand_cont = torch.zeros(*hand_model_params.shape[:-1], 54).to(hand_model_params)
    hand_cont[..., mask_cont_threedofs] = hand_cont_threedofs
    hand_cont[..., mask_cont_onedofs] = hand_cont_onedofs

    return hand_cont

def batch9Dfrom6D(poses):


    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1).flatten(-2, -1)

    return matrix

def batch4Dfrom2D(poses):

    poses_norm = F.normalize(poses, dim=-1)

    poses_4d = torch.stack(
        [
            poses_norm[..., 1],
            poses_norm[..., 0],
            -poses_norm[..., 0],
            poses_norm[..., 1],
        ],
        dim=-1,
    )

    return poses_4d

def compact_cont_to_rotmat_body(body_pose_cont, inflate_trans=False):

    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])

    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )

    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

    * First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).flatten(-2, -1)
    * Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))
    body_rotmat_1dofs = batch4Dfrom2D(body_cont_1dofs).flatten(-2, -1)
    if inflate_trans:
        assert False, (
            "This is left as a possibility to increase the space/contribution/supervision trans params gets compared to rots"
        )
    else:
        * Nothing to do for trans
        body_rotmat_trans = body_cont_trans

    body_rotmat_params = torch.cat(
        [body_rotmat_3dofs, body_rotmat_1dofs, body_rotmat_trans], dim=-1
    )
    return body_rotmat_params

def compact_cont_to_model_params_body(body_pose_cont):

    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])

    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )

    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

    * First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_params_3dofs = batchXYZfrom6D(body_cont_3dofs).flatten(-2, -1)
    * Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])
    * Nothing to do for trans
    body_params_trans = body_cont_trans

    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)
    body_pose_params[..., all_param_3dof_rot_idxs.flatten()] = body_params_3dofs
    body_pose_params[..., all_param_1dof_rot_idxs] = body_params_1dofs
    body_pose_params[..., all_param_1dof_trans_idxs] = body_params_trans
    return body_pose_params

def compact_model_params_to_cont_body(body_pose_params):

    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])

    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_params.shape[-1] == (
        num_3dof_angles + num_1dof_angles + num_1dof_trans
    )

    body_params_3dofs = body_pose_params[..., all_param_3dof_rot_idxs.flatten()]
    body_params_1dofs = body_pose_params[..., all_param_1dof_rot_idxs]
    body_params_trans = body_pose_params[..., all_param_1dof_trans_idxs]

    body_cont_3dofs = batch6DFromXYZ(body_params_3dofs.unflatten(-1, (-1, 3))).flatten(
        -2, -1
    )
    body_cont_1dofs = torch.stack(
        [body_params_1dofs.sin(), body_params_1dofs.cos()], dim=-1
    ).flatten(-2, -1)
    body_cont_trans = body_params_trans

    body_pose_cont = torch.cat(
        [body_cont_3dofs, body_cont_1dofs, body_cont_trans], dim=-1
    )
    return body_pose_cont

mhr_param_hand_idxs = [62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
mhr_cont_hand_idxs = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237]
mhr_param_hand_mask = torch.zeros(133).bool(); mhr_param_hand_mask[mhr_param_hand_idxs] = True
mhr_cont_hand_mask = torch.zeros(260).bool(); mhr_cont_hand_mask[mhr_cont_hand_idxs] = True

```

* Файл: sam_3d_body/models/modules/misc.py

```python

import collections.abc
from itertools import repeat

def _ntuple(n):
    """A `to_tuple` function generator.

    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

```

* Файл: sam_3d_body/models/modules/swiglu_ffn.py

```python

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath

from .layer_scale import LayerScale

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)

        self.norm = norm_layer

        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma2 = nn.Identity()

        self.dropout_layer = DropPath(drop_path_rate)
        self.add_identity = add_identity

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:


            return out

        if identity is None:
            identity = x
        return identity + out

class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            layer_scale_init_value=layer_scale_init_value,
            bias=bias,
        )

```

* Файл: sam_3d_body/models/modules/transformer.py

```python

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath

from .layer_scale import LayerScale
from .swiglu_ffn import SwiGLUFFNFused

class MLP(nn.Module):

    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

def build_norm_layer(cfg: Dict, num_features: int):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type == "LN":
        norm_layer = LayerNorm32
    else:
        raise ValueError("Unsupported norm layer: ", layer_type)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.ReLU
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """







    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        output_dims=None,
        num_fcs=2,
        act_layer=nn.ReLU,
        ffn_drop=0.0,
        drop_path_rate=0.0,
        add_identity=True,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_dims = output_dims or embed_dims
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    act_layer(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(in_channels, self.output_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()
        )
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()


    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class MultiheadAttention(nn.Module):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dims)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

class Attention(nn.Module):
    """Multi-head Attention Module for both self and cross attention.

    Support masking invalid elements for attention.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        query_dims=None,
        key_dims=None,
        value_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()

        self.query_dims = query_dims or embed_dims
        self.key_dims = key_dims or embed_dims
        self.value_dims = value_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.q_proj = nn.Linear(self.query_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(self.key_dims, embed_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(self.value_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, self.query_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dims)
        return x.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = q.shape
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        attn_drop = self.attn_drop if self.training else 0.0
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=attn_drop
        )
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        layer_scale_init_value=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        ffn_type="origin",
        act_layer=nn.GELU,
        norm_cfg=dict(type="LN", eps=1e-6),
    ):
        super().__init__()

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x

class TransformerDecoderLayer(nn.Module):
    """Implements one decoder layer in cross-attention Transformer.

    Adapted from Segment Anything Model (SAM) implementation.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        enable_twoway (bool): Whether to enable two-way Transformer (used in SAM).
        repeat_pe (bool): Whether to re-add PE at each layer (used in SAM)
        skip_first_pe (bool)
    """

    def __init__(
        self,
        token_dims: int,
        context_dims: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        skip_first_pe: bool = False,
    ):
        super().__init__()
        self.repeat_pe = repeat_pe
        self.skip_first_pe = skip_first_pe
        if self.repeat_pe:
            self.ln_pe_1 = build_norm_layer(norm_cfg, token_dims)
            self.ln_pe_2 = build_norm_layer(norm_cfg, context_dims)

        self.ln1 = build_norm_layer(norm_cfg, token_dims)

        self.self_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=token_dims,
            value_dims=token_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln2_1 = build_norm_layer(norm_cfg, token_dims)
        self.ln2_2 = build_norm_layer(norm_cfg, context_dims)

        self.cross_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=context_dims,
            value_dims=context_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.ln3 = build_norm_layer(norm_cfg, token_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            raise NotImplementedError

        self.enable_twoway = enable_twoway
        if self.enable_twoway:
            self.ln4_1 = build_norm_layer(norm_cfg, context_dims)
            self.ln4_2 = build_norm_layer(norm_cfg, token_dims)

            self.cross_attn_2 = Attention(
                embed_dims=num_heads * head_dims,
                num_heads=num_heads,
                query_dims=context_dims,
                key_dims=token_dims,
                value_dims=token_dims,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        x_pe: Optional[torch.Tensor] = None,
        context_pe: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: shape [B, N, C]
            context: shape [B, N, C]
            x_mask: shape [B, N]
        """
        if self.repeat_pe and context_pe is not None:

            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)


        if self.repeat_pe and not self.skip_first_pe and x_pe is not None:
            q = k = self.ln1(x) + x_pe
            v = self.ln1(x)
        else:
            q = k = v = self.ln1(x)

        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask[:, :, None] @ x_mask[:, None, :]

            attn_mask.diagonal(dim1=1, dim2=2).fill_(1)
            attn_mask = attn_mask > 0
        x = x + self.self_attn(q=q, k=k, v=v, attn_mask=attn_mask)


        if self.repeat_pe and context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q=q, k=k, v=v)


        x = self.ffn(self.ln3(x), identity=x)


        if self.enable_twoway:
            if self.repeat_pe and context_pe is not None:
                q = self.ln4_1(context) + context_pe
                k = self.ln4_2(x) + x_pe
                v = self.ln4_2(x)
            else:
                q = self.ln4_1(context)
                k = v = self.ln4_2(x)
            attn_mask = (
                (x_mask[:, None, :].repeat(1, context.shape[1], 1)) > 0
                if x_mask is not None
                else None
            )
            context = context + self.cross_attn_2(q=q, k=k, v=v, attn_mask=attn_mask)

        return x, context

```

* Файл: sam_3d_body/models/optim/fp16_utils.py

```python

import torch
import torch.nn as nn

FP16_MODULES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
]

FP16_MODULES = tuple(FP16_MODULES)

def convert_to_fp16_safe(module, dtype=torch.float16):
    for child in module.children():
        convert_to_fp16_safe(child, dtype)
    if not isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        module.to(dtype)

def convert_module_to_f16(l, dtype=torch.float16):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():

            p.data = p.data.to(dtype)

def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

```

* Файл: sam_3d_body/models/optim/__init__.py

```python

```

* Файл: sam_3d_body/sam_3d_body_estimator.py

```python

from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor

class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4


        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """


        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []


        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False


        if len(boxes) == 0:
            return []


        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        masks_score = None
        if masks is not None:

            print(f"Using provided masks: {masks.shape}")
            assert bboxes is not None, (
                "Mask-conditioned inference requires bboxes input!"
            )
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")

            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None


        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)


        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)



        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

```

* Файл: sam_3d_body/utils/checkpoint.py

```python

from collections import namedtuple

import pytorch_lightning as pl
import torch

from .logging import get_pylogger

log = get_pylogger(__name__)

class CheckpointCallback(pl.callbacks.ModelCheckpoint):
    """Disable model checkpoint after validation to avoid DDP job hanging after resume"""

    def on_validation_end(self, trainer, pl_module):

        pass

class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []


    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata


    def load(module, local_state_dict, prefix=""):


        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)


        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, "_load_state_dict_post_hooks"):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with "
                    "``register_load_state_dict_post_hook`` are not expected "
                    "to return new values, if incompatible_keys need to be "
                    "modified, it should be done inplace."
                )

    load(module, state_dict)
    load = None


    missing_keys = [key for key in missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n"
        )
    if missing_keys:
        err_msg.append(
            f"missing keys in source state_dict: {', '.join(missing_keys)}\n"
        )

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            log.warning(err_msg)

```

* Файл: sam_3d_body/utils/config.py

```python

from typing import Dict

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from yacs.config import CfgNode as CN

def resolve_omegaconf_to_dict(conf):
    """
    Recursively convert an OmegaConf object to a dictionary, resolving interpolations
    where possible and leaving unsupported ones as-is.
    """
    if isinstance(conf, DictConfig):
        result = {}
        for k, v in conf.items():
            try:
                result[k] = resolve_omegaconf_to_dict(v)
            except InterpolationResolutionError:

                result[k] = OmegaConf.to_container(v, resolve=False)
        return result
    elif isinstance(conf, ListConfig):
        result = []
        for item in conf:
            try:
                result.append(resolve_omegaconf_to_dict(item))
            except InterpolationResolutionError:

                result.append(OmegaConf.to_container(item, resolve=False))
        return result
    else:

        if OmegaConf.is_config(conf):
            try:
                return OmegaConf.to_container(conf, resolve=True)
            except InterpolationResolutionError:

                return OmegaConf.to_container(conf, resolve=False)
        else:

            return conf

def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}

def get_config(config_file: str) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)


    conf = OmegaConf.load(config_file)
    conf_dict = resolve_omegaconf_to_dict(conf)
    conf_cfg = CN(conf_dict)


    cfg.merge_from_other_cfg(conf_cfg)

    cfg.freeze()
    return cfg

```

* Файл: sam_3d_body/utils/dist.py

```python

import os
import pickle
import shutil
import tempfile
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import distributed as torch_dist, Tensor
from torch.distributed import ProcessGroup

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        if target == "numpy":
            return x.numpy()
        else:
            return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x

def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()

def get_default_group():
    """Return default process group."""
    return torch_dist.distributed_c10d._get_default_group()

def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():


        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1

def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():


        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0

def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size

def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0

def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():


        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)

def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>>
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>>
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>>
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        "device type in data should be consistent, but got "
                        f"{cur} and {pre}"
                    )
        if pre is None:
            raise ValueError("data should not be empty.")
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        "device type in data should be consistent, but got "
                        f"{cur} and {pre}"
                    )
        if pre is None:
            raise ValueError("data should not be empty.")
        return pre
    else:
        raise TypeError(
            f"data should be a Tensor, sequence of tensor or dict, but got {data}"
        )

def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():


        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None

def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == "hccl":
        import torch_npu

        return torch.device("npu", torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device("cuda", torch.cuda.current_device())
    elif backend == "cncl":
        import torch_mlu

        return torch.device("mlu", torch.mlu.current_device())
    elif backend == "smddp":
        return torch.device("cuda", torch.cuda.current_device())
    else:

        return torch.device("cpu")

def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None,
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                "out should be the same type with data, but got data is "
                f"{type(data)} and out is {type(data)}"
            )

        if isinstance(out, set):
            raise TypeError("out should not be a set")

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:

            out.copy_(data_on_device)

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)
            if data_len != out_len:
                raise ValueError(
                    "length of data and out should be same, "
                    f"but got {data_len} and {out_len}"
                )

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device, out[k])
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError("data should not be empty")



        return type(data)(data_on_device)
    elif (
        isinstance(data, Iterable)
        and not isinstance(data, str)
        and not isinstance(data, np.ndarray)
    ):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError("data should not be empty")

        return type(data)(data_on_device)
    else:
        raise TypeError(
            f"data should be a Tensor, list of tensor or dict, but got {data}"
        )

def broadcast(data: Tensor, src: int = 0, group: Optional[ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>>
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>>
        >>>
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2])
        tensor([3, 4])
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2])
        tensor([1, 2])
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)

        data_on_device = data_on_device.contiguous()
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)

def broadcast_object_list(
    data: List[Any], src: int = 0, group: Optional[Any] = None
) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Note:
        Calling ``broadcast_object_list`` in non-distributed environment does
        nothing.

    Args:
        data (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank
            will be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Note:
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>>
        >>> data = ['foo', 12, {1: 2}]
        >>> dist.broadcast_object_list(data)
        >>> data
        ['foo', 12, {1: 2}]

        >>>
        >>>
        >>> if dist.get_rank() == 0:
        >>>
        >>>     data = ["foo", 12, {1: 2}]
        >>> else:
        >>>     data = [None, None, None]
        >>> dist.broadcast_object_list(data)
        >>> data
        ["foo", 12, {1: 2}]
        ["foo", 12, {1: 2}]
    """
    assert isinstance(data, list)

    if get_world_size() > 1:
        if group is None:
            group = get_default_group()

        torch_dist.broadcast_object_list(data, src, group)

def collect_results(
    results: list, size: int, device: str = "cpu", tmpdir: Optional[str] = None
) -> Optional[list]:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu', 'gpu' or 'npu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu' or 'npu'.
            Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>>
        >>>
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]
        None
    """
    if device not in ["gpu", "cpu", "npu"]:
        raise NotImplementedError(
            f"device must be 'cpu' , 'gpu' or 'npu', but got {device}"
        )

    if device == "gpu" or device == "npu":
        return _collect_results_device(results, size)
    else:
        return collect_results_cpu(results, size, tmpdir)

def _collect_results_device(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu or npu mode."""
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]



    part_list = [None] * world_size
    group = get_default_group()
    torch_dist.all_gather_object(part_list, result_part, group)

    if rank == 0:

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None

def collect_results_cpu(
    result_part: list, size: int, tmpdir: Optional[str] = None
) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>>
        >>>
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]
        None
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]


    if tmpdir is None:
        MAX_LEN = 512

        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8)
        if rank == 0:
            os.makedirs(".dist_test", exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[: len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)


    with open(os.path.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(result_part, f, protocol=2)

    barrier()

    if rank != 0:
        return None
    else:

        part_list = []
        for i in range(world_size):
            path = os.path.join(tmpdir, f"part_{i}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{tmpdir} is not an shared directory for "
                    f"rank {i}, please make sure {tmpdir} is a shared "
                    "directory for all ranks!"
                )
            with open(path, "rb") as f:
                part_list.append(pickle.load(f))






        part_list = [single for single in part_list if len(single) > 0]

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]

        shutil.rmtree(tmpdir)
        return ordered_results

```

* Файл: sam_3d_body/utils/__init__.py

```python

from .dist import recursive_to

```

* Файл: sam_3d_body/utils/logging.py

```python

import logging

from pytorch_lightning.utilities import rank_zero_only

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)



    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

```

* Файл: sam_3d_body/visualization/__init__.py

```python

```

* Файл: sam_3d_body/visualization/renderer.py

```python

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
from typing import List, Optional

import cv2
import numpy as np
import pyrender
import torch
import trimesh

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):

    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )

    return nodes

class Renderer:
    def __init__(self, focal_length, faces=None):
        """
        Wrapper around the pyrender renderer to render meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        self.focal_length = focal_length
        self.faces = faces

    def __call__(
        self,
        vertices: np.array,
        cam_t: np.array,
        image: np.ndarray,
        full_frame: bool = False,
        imgname: Optional[str] = None,
        side_view=False,
        top_view=False,
        rot_angle=90,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        tri_color_lights=False,
        return_rgba=False,
        camera_center=None,
    ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            cam_t (np.array): Array of shape (3,) with the camera translation.
            image (np.array): Array of (H, W, 3) containing the cropped image (unnormalized values).
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """

        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)
        image = image / 255.0
        h, w = image.shape[:2]

        renderer = pyrender.OffscreenRenderer(
            viewport_height=h,
            viewport_width=w,
        )

        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(
                mesh_base_color[2],
                mesh_base_color[1],
                mesh_base_color[0],
                1.0,
            ),
        )
        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0]
            )
            mesh.apply_transform(rot)
        elif top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0]
            )
            mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        if camera_center is None:
            camera_center = [image.shape[1] / 2.0, image.shape[0] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        if tri_color_lights:
            colors = [
                np.array([1, 0.2, 0.3]),
                np.array([0.2, 1, 0.2]),
                np.array([0.2, 0.2, 1]),
            ]
            for ln, color in zip(light_nodes, colors):
                ln.light.color = color
                ln.light.intensity = 2.0

        for node in light_nodes:
            scene.add_node(node)

        color, _rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(
        self,
        vertices,
        camera_translation,
        mesh_base_color=(1.0, 1.0, 0.9),
        rot_axis=[1, 0, 0],
        rot_angle=0,
    ):




        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(
            vertices.copy() + camera_translation,
            self.faces.copy(),
            vertex_colors=vertex_colors,
        )



        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
        self,
        vertices: np.array,
        cam_t=None,
        rot=None,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        camera_z=3,

        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
    ):
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0
        )





        if cam_t is not None:
            camera_translation = cam_t.copy()

        else:
            camera_translation = np.array(
                [0, 0, camera_z * self.focal_length / render_res[1]]
            )

        mesh = self.vertices_to_trimesh(
            vertices, camera_translation, mesh_base_color, rot_axis, rot_angle
        )
        mesh = pyrender.Mesh.from_trimesh(mesh)


        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)

        camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )


        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_rgba_multiple(
        self,
        vertices: List[np.array],
        cam_t: List[np.array],
        rot_axis=[1, 0, 0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
        focal_length=None,
    ):
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0
        )
        MESH_COLORS = [
            [0.000, 0.447, 0.741],
            [0.850, 0.325, 0.098],
            [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556],
            [0.466, 0.674, 0.188],
            [0.301, 0.745, 0.933],
        ]
        mesh_list = [
            pyrender.Mesh.from_trimesh(
                self.vertices_to_trimesh(
                    vvv,
                    ttt.copy(),
                    MESH_COLORS[n % len(MESH_COLORS)],
                    rot_axis,
                    rot_angle,
                )
            )
            for n, (vvv, ttt) in enumerate(zip(vertices, cam_t))
        ]

        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f"mesh_{i}")

        camera_pose = np.eye(4)

        camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )


        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):

        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):

        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose





            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

```

* Файл: sam_3d_body/visualization/skeleton_visualizer.py

```python

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from .utils import draw_text, parse_pose_metainfo

class SkeletonVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight


        self.pose_meta = {}
        self.skeleton = None

    def set_pose_meta(self, pose_meta: Dict):
        parsed_meta = parse_pose_metainfo(pose_meta)

        self.pose_meta = parsed_meta.copy()
        self.bbox_color = parsed_meta.get("bbox_color", self.bbox_color)
        self.kpt_color = parsed_meta.get("keypoint_colors", self.kpt_color)
        self.link_color = parsed_meta.get("skeleton_link_colors", self.link_color)
        self.skeleton = parsed_meta.get("skeleton_links", self.skeleton)

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 3
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]


        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-1]
            score = cur_keypoints[:, -1]

            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color "
                    f"({len(self.kpt_color)}) does not matches "
                    f"that of keypoints ({len(kpts)})"
                )


            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) does not matches "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):

                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        color,
                        thickness=self.line_width,
                    )


            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:

                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    kpt[0] += self.radius
                    kpt[1] -= self.radius
                    image = draw_text(
                        image,
                        str(kid),
                        kpt,
                        image_size=(img_w, img_h),
                        color=color,
                        font_size=self.radius * 3,
                        vertical_alignment="bottom",
                        horizontal_alignment="center",
                    )

        return image

    def draw_skeleton_analysis(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of prediction.
        The color is determined by whether the keypoint is correctly predicted.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 4
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]


        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-2]
            score = cur_keypoints[:, -2]
            correct = cur_keypoints[:, -1]

            kpt_color = [
                [0, 255, 0] if correct[kid] else [0, 0, 255] for kid in range(len(kpts))
            ]
            kpt_color = np.array(kpt_color, dtype=np.uint8)


            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) does not matches "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):

                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        color,
                        thickness=self.line_width,
                    )


            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:

                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    kpt[0] += self.radius
                    kpt[1] -= self.radius
                    image = draw_text(
                        image,
                        str(kid),
                        kpt,
                        image_size=(img_w, img_h),
                        color=color,
                        font_size=self.radius * 3,
                        vertical_alignment="bottom",
                        horizontal_alignment="center",
                    )

        return image

```

* Файл: sam_3d_body/visualization/utils.py

```python

import os
from typing import Dict, Optional, Union

import cv2
import numpy as np
from detectron2.config import LazyConfig
from omegaconf import OmegaConf

def draw_text(
    image: np.ndarray,
    texts: str,
    positions: np.ndarray,
    image_size: Optional[tuple] = None,
    font_size: Optional[int] = None,
    color: Union[str, tuple] = "g",
    vertical_alignment: str = "top",
    horizontal_alignment: str = "left",
):
    """Draw single or multiple text boxes.

    Args:
        texts (Union[str, List[str]]): Texts to draw.
        positions (np.ndarray: The position to draw
            the texts, which should have the same length with texts and
            each dim contain x and y.
        image_size (Optional[tuple]): image size to bound text drawing.
            (width, height)
        font_size (Union[int, List[int]], optional): The font size of
            texts.  Defaults to None.
        color (Union[str, tuple): The colors of texts.
        vertical_alignment (str): The verticalalignment
            of texts. verticalalignment controls whether the y positional
            argument for the text indicates the bottom, center or top side
            of the text bounding box.
        horizontal_alignment (str): The
            horizontalalignment of texts. Horizontalalignment controls
            whether the x positional argument for the text indicates the
            left, center or right side of the text bounding box.
    """
    font_scale = max(0.1, font_size / 30)
    thickness = max(1, font_size // 15)

    text_size, text_baseline = cv2.getTextSize(
        texts, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )

    x = int(positions[0])
    if horizontal_alignment == "right":
        x = max(0, x - text_size[0])
    y = int(positions[1])
    if vertical_alignment == "top":
        y = y + text_size[1]
        if image_size is not None:
            y = min(image_size[1], y)

    return cv2.putText(
        image, texts, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness - 1
    )

def draw_box(
    img,
    bbox=[],
    text="",
    box_color=(0, 255, 0),
    text_color=(0, 255, 0),
    font_scale=0.7,
    font_thickness=1,
):

    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    img = cv2.rectangle(
        img,
        pt1,
        pt2,
        box_color,
        2,
    )
    if text:
        y, dy = int(bbox[1]) + 30, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_origin = (pt1[0] + 2, pt1[1] + text_size[1] + 2)
        for line in text.split("\n"):
            img = cv2.putText(
                img,
                str(line),
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            y += dy

    return img

def parse_pose_metainfo(metainfo: Union[str, Dict]):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "pose_format" (str): The name of the pose format
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "pose_format": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }

            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = mmengine.Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """

    if type(metainfo) == str:
        if not os.path.isfile(metainfo):
            raise ValueError("Invalid metainfo file path: ", metainfo)
        metainfo = OmegaConf.to_container(LazyConfig.load(metainfo).pose_info)


    assert "pose_format" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo
    assert "joint_weights" in metainfo
    assert "sigmas" in metainfo


    parsed = dict(
        pose_format=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["pose_format"] = metainfo["pose_format"]

    if "remove_teeth" in metainfo:
        parsed["remove_teeth"] = metainfo["remove_teeth"]

    if "min_visible_keypoints" in metainfo:
        parsed["min_visible_keypoints"] = metainfo["min_visible_keypoints"]

    if "teeth_keypoint_ids" in metainfo:
        parsed["teeth_keypoint_ids"] = metainfo["teeth_keypoint_ids"]

    if "coco_wholebody_to_goliath_mapping" in metainfo:
        parsed["coco_wholebody_to_goliath_mapping"] = metainfo[
            "coco_wholebody_to_goliath_mapping"
        ]

    if "coco_wholebody_to_goliath_keypoint_info" in metainfo:
        parsed["coco_wholebody_to_goliath_keypoint_info"] = metainfo[
            "coco_wholebody_to_goliath_keypoint_info"
        ]


    parsed["num_keypoints"] = len(metainfo["keypoint_info"])

    for kpt_id, kpt in metainfo["keypoint_info"].items():
        kpt_name = kpt["name"]
        parsed["keypoint_id2name"][kpt_id] = kpt_name
        parsed["keypoint_name2id"][kpt_name] = kpt_id
        parsed["keypoint_colors"].append(kpt.get("color", [255, 128, 0]))

        kpt_type = kpt.get("type", "")
        if kpt_type == "upper":
            parsed["upper_body_ids"].append(kpt_id)
        elif kpt_type == "lower":
            parsed["lower_body_ids"].append(kpt_id)

        swap_kpt = kpt.get("swap", "")
        if swap_kpt == kpt_name or swap_kpt == "":
            parsed["flip_indices"].append(kpt_name)
        else:
            parsed["flip_indices"].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed["flip_pairs"]:
                parsed["flip_pairs"].append(pair)


    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for _, sk in metainfo["skeleton_info"].items():
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [96, 96, 255]))


    parsed["dataset_keypoint_weights"] = np.array(
        metainfo["joint_weights"], dtype=np.float32
    )
    parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)

    if "stats_info" in metainfo:
        parsed["stats_info"] = {}
        for name, val in metainfo["stats_info"].items():
            parsed["stats_info"][name] = np.array(val, dtype=np.float32)


    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed["flip_pairs"] = _map(
        parsed["flip_pairs"], mapping=parsed["keypoint_name2id"]
    )
    parsed["flip_indices"] = _map(
        parsed["flip_indices"], mapping=parsed["keypoint_name2id"]
    )
    parsed["skeleton_links"] = _map(
        parsed["skeleton_links"], mapping=parsed["keypoint_name2id"]
    )

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(
        parsed["skeleton_link_colors"], dtype=np.uint8
    )

    return parsed

```

* Файл: tools/build_detector.py

```python

import os
from pathlib import Path

import numpy as np
import torch

class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device

        if name == "vitdet":
            print("
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        else:
            raise NotImplementedError

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)

def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig


    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector

def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )
    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()


    sorted_indices = np.lexsort(
        (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    )
    boxes = boxes[sorted_indices]
    return boxes

```

* Файл: tools/build_fov_estimator.py

```python

import torch

class FOVEstimator:
    def __init__(self, name="moge2", device="cuda", **kwargs):
        self.device = device

        if name == "moge2":
            print("
            self.fov_estimator = load_moge(device, **kwargs)
            self.fov_estimator_func = run_moge

            self.fov_estimator.eval()
        else:
            raise NotImplementedError

    def get_cam_intrinsics(self, img, **kwargs):
        return self.fov_estimator_func(self.fov_estimator, img, self.device, **kwargs)

def load_moge(device, path=""):
    from moge.model.v2 import MoGeModel

    if path == "":
        path = "Ruicheng/moge-2-vitl-normal"
    moge_model = MoGeModel.from_pretrained(path).to(device)
    return moge_model

def run_moge(model, input_image, device):

    H, W, _ = input_image.shape
    input_image = torch.tensor(
        input_image / 255, dtype=torch.float32, device=device
    ).permute(2, 0, 1)


    moge_data = model.infer(input_image)


    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]


    intrinsics[0, 0] = v_focal

    cam_intrinsics = intrinsics[None]

    return cam_intrinsics

def denormalize_f(norm_K, height, width):

    cx_norm = norm_K[0][2]
    cy_norm = norm_K[1][2]

    fx_norm = norm_K[0][0]
    fy_norm = norm_K[1][1]



    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height

    s_abs = 0


    abs_K = torch.tensor(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]]
    )
    return abs_K

```

* Файл: tools/build_sam.py

```python

import torch
import numpy as np

class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2

        else:
            raise NotImplementedError

    def run_sam(self, img, boxes, **kwargs):
        return self.sam_func(self.sam, img, boxes)

def load_sam2(device, path):
    checkpoint = f"{path}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    import sys

    sys.path.append(path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.eval()

    return predictor

def run_sam2(sam_predictor, img, boxes):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img)
        all_masks, all_scores = [], []
        for i in range(boxes.shape[0]):

            masks, scores, logits = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[[i]],
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            mask_1 = masks[0]
            score_1 = scores[0]
            all_masks.append(mask_1)
            all_scores.append(score_1)


        all_masks = np.stack(all_masks)
        all_scores = np.stack(all_scores)

    return all_masks, all_scores

```

* Файл: tools/cascade_mask_rcnn_vitdet_h_75ep.py

```python

* coco_loader_lsj.py

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

image_size = 1024
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 64

dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]

from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"

train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

from functools import partial

train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_huge_p14to16.pth"
)

model.backbone.net.embed_dim = 1280
model.backbone.net.depth = 32
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.5

model.backbone.net.window_block_indexes = (
    list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
)

optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=32
)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = train.max_iter * 3 // 4
lr_multiplier.scheduler.milestones = [
    milestone * 3 // 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter

```

* Файл: tools/__init__.py

```python

```

* Файл: tools/vis_utils.py

```python

import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)

def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):

    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()


    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]


    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)


    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(
            person_output["pred_vertices"] + person_output["pred_cam_t"]
        )
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)


    fake_pred_cam_t = (
        np.max(all_pred_vertices[-2 * 18439 :], axis=0)
        + np.min(all_pred_vertices[-2 * 18439 :], axis=0)
    ) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t


    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )


    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        * 255
    )

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img

```
