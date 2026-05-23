import math
import numpy as np
import torch
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_TARGET_RATIOS_CACHE = {}

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def get_target_ratios(min_num=1, max_num=12):
    key = (min_num, max_num)
    if key not in _TARGET_RATIOS_CACHE:
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        _TARGET_RATIOS_CACHE[key] = sorted(target_ratios, key=lambda x: x[0] * x[1])
    return _TARGET_RATIOS_CACHE[key]

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = get_target_ratios(min_num=min_num, max_num=max_num)

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def dynamic_preprocess_parallel(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = get_target_ratios(min_num=min_num, max_num=max_num)
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        resized_future = executor.submit(image.resize, (target_width, target_height))
        thumbnail_future = (
            executor.submit(image.resize, (image_size, image_size))
            if use_thumbnail and blocks != 1
            else None
        )
        resized_img = resized_future.result()
        thumbnail_img = thumbnail_future.result() if thumbnail_future is not None else None

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if thumbnail_img is not None:
        processed_images.append(thumbnail_img)
    return processed_images

def normalize_numpy_images(images):
    array = np.stack(images, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std

def load_image_opencv(image_file, input_size=448, max_num=12):
    import cv2

    image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_file}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    aspect_ratio = orig_width / orig_height
    target_ratios = get_target_ratios(min_num=1, max_num=max_num)
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, input_size)

    target_width = input_size * target_aspect_ratio[0]
    target_height = input_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )
    processed_images = []
    grid_width = target_width // input_size
    for i in range(blocks):
        left = (i % grid_width) * input_size
        top = (i // grid_width) * input_size
        processed_images.append(resized[top:top + input_size, left:left + input_size])

    if blocks != 1:
        thumbnail = cv2.resize(
            image,
            (input_size, input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        processed_images.append(thumbnail)

    return normalize_numpy_images(processed_images)

def load_image(image_file, input_size=448, max_num=12, backend="pil"):
    backend = backend.lower()
    if backend in {"opencv", "cv2"}:
        return load_image_opencv(image_file, input_size=input_size, max_num=max_num)

    if backend == "pil_draft":
        image = Image.open(image_file)
        image.draft("RGB", (input_size * max_num, input_size * max_num))
        image = image.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    transform = build_transform(input_size=input_size)
    if backend == "pil_parallel":
        images = dynamic_preprocess_parallel(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    elif backend in {"pil", "pil_draft"}:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        raise ValueError(f"Unsupported image preprocessing backend: {backend}")
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
