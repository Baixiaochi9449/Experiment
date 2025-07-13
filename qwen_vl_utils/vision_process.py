from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MAX_RATIO = 200

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 128 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 16

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'. 宽和高要能整除factor(为了后面划分token方便,每个token对应的h 和 w 为 factor:28)

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels']. 像素总数量要在规定的['min_pixels', 'max_pixels']范围内

    3. The aspect ratio of the image is maintained as closely as possible. 尽可能保持横纵比
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor)) # round_by_factor:求出能整除factor且与 h/w 的最接近的值
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels: # 如果此时像素值大于max_pixels 则让调整h_bar w_bar
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor) # 需要满足整除factor
        w_bar = floor_by_factor(width / beta, factor) # 由于是向下取整，可以保证h_bar * w_bar <= max_pixels
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor) # 需要满足整除factor
        w_bar = ceil_by_factor(width * beta, factor) # 由于是向下取整，可以保证h_bar * w_bar >= min_pixels
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"] # 图片地址 '/home/gwj/Video-R1/src/r1-v/Video-R1-data/Knowledge/ArxivQA/images/2109.07509_0.jpg'
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image) # 读取图片
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj) # 将图片转为RGB模式
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size # 960 540
        min_pixels = ele.get("min_pixels", MIN_PIXELS) # 单张图像的最小像素值
        max_pixels = ele.get("max_pixels", MAX_PIXELS) # 单张图像的最大像素值
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )# 根据像素值取值区间且能整除size_factor确定需要调整之后的图像宽 高
    image = image.resize((resized_width, resized_height)) # 改变图像大小

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS) # 采样之后的视频帧的目标fps
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR) # 确定最小的目标frames数量,为什么这个数量必须能整除FRAME_FACTOR?
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR) # 确定最大的目标frames数量
        nframes = total_frames / video_fps * fps # 按照已定的FPS，计算所需的frames数量
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames) # 对于nframes又以下要求 1.不能超过total_frames 2.不能超过max_frames 3.不能少于min_frames
        nframes = floor_by_factor(nframes, FRAME_FACTOR) # 由于以上限制最终确定nframes = 16
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"] # video_path: /home/gwj/Video-R1/src/r1-v/Video-R1-data/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_1P7c8f5K1Xw.mp4
    st = time.time() # 当前时间
    vr = decord.VideoReader(video_path) # 对视频进行读取并解码 len(vr) = 1703;vr[i]为第i帧的视频对应的ndarray矩阵;vr[0].shape:(640,360,3)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps() # 1703 30.0
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps) # nframes的要求: 1.根据目标FPS，求出应该采样的帧数nframse 2. nframes的取值范围在设定的[min_frames,max_frame],如果1中求出的nframes < min_frames则取min_frames为采样数，如果1中求出的nframes > max_frames则取max_frames为采样数
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist() # 均匀选取16帧 idx = [0, 113, 227, 340, 454, 567, 681, 794, 908, 1021, 1135, 1248, 1362, 1475, 1589, 1702]
    video = vr.get_batch(idx).asnumpy()# shape为(16,640,360,3) 筛选出下标为idx对应的帧
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format video.shape = torch.Size([16, 3, 640, 360])
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps # 采样了16帧，计算平均帧率
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str): # 是否是字符串类型
        video_reader_backend = get_video_reader_backend() # 选定对应的视频解码函数
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele) # 调用_read_video_decord函数进行视频采样与解析,_read_video_decord函数需要理解
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
        
        # 这些代码是计算视频的最大像素限制。根据传入的参数 min_pixels、total_pixels 等来确定 max_pixels 的值，用于调整视频大小的限制。
        nframes, _, height, width = video.shape # 16,_,640,360
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS) # 最小像素总量
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05)) # 最大像素总量
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        # min_pixels = 100352 max_pixels_supposed = max_pixels = 105369
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )# 根据像素限制调整高度和宽度 resized_height,resized_width = 420,224
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()#使用 transforms.functional.resize 来调整视频的尺寸，这里使用了双三次插值（BICUBIC）和抗锯齿（antialias=True）技术来进行图像的平滑处理
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
    # 如果是图片的话: [{'type': 'image', 'image': '/home/gwj/Video-R1/src/r1-v/Video-R1-data/Knowledge/ArxivQA/images/2109.07509_0.jpg'}]
    vision_infos = extract_vision_info(conversations) # 提取出视频信息 [{'type': 'video', 'video': '/home/gwj/Video-R1/src/r1-v/Video-R1-data/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_1P7c8f5K1Xw.mp4'}]
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos: # 遍历多个图片/视频(推理的时候就有多个了)
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True) # 提取出采样之后的视频内容和帧数 video_input:[16,3,w,h] fetch_video函数需要理解
            video_sample_fps_list.append(video_sample_fps) # 记录fps
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
