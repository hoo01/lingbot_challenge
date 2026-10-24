"""Offline GroundingDINO box proposals in the reader's canonical coordinates.

No detector import/forward is performed by a training worker. Detection confidence
is not a semantic review: all produced Foreground objects stay unreviewed until an
independent, RGB/SHA-bound review approves them. Missing objects never become
trusted background merely because another object's box was detected.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .foreground import Foreground, array_sha, file_sha


@dataclass(frozen=True)
class BoxPolicy:
    score_threshold: float = .25
    text_threshold: float = .20
    minimum_margin_pixels: int = 4
    relative_margin: float = .10
    # images.py additionally applies its existing 7-pixel protection halo.
    augmentation_halo_pixels: int = 7

    def __post_init__(self):
        if not (0 < self.score_threshold < 1 and 0 < self.text_threshold < 1
                and 0 <= self.minimum_margin_pixels <= 32
                and 0 <= self.relative_margin <= .5
                and self.augmentation_halo_pixels == 7):
            raise ValueError('Invalid recorded box protection policy')


def expanded_box_mask(boxes, shape=(256, 256), *, policy=BoxPolicy()):
    """XYXY pixels, floor left/top and ceil right/bottom, clipped to the image.

    Invalid/degenerate boxes are errors, never silently discarded. Empty results
    produce an empty UNREVIEWED proposal, not permission to erase the image.
    """
    if shape != (256, 256):
        raise ValueError('Boxes must be in canonical 256x256 head-camera coordinates')
    a = np.asarray(boxes, dtype=np.float64)
    if a.size == 0:
        a = a.reshape(0, 4)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError('Finite XYXY [n,4] boxes required')
    mask = np.zeros(shape, dtype=bool)
    expanded = []
    for x1, y1, x2, y2 in a:
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= 256 or y1 >= 256:
            raise ValueError('Degenerate/outside detector box')
        dx = max(policy.minimum_margin_pixels, policy.relative_margin * (x2 - x1))
        dy = max(policy.minimum_margin_pixels, policy.relative_margin * (y2 - y1))
        left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
        right, bottom = min(256, math.ceil(x2 + dx)), min(256, math.ceil(y2 + dy))
        mask[top:bottom, left:right] = True
        expanded.append([left, top, right, bottom])
    return mask, expanded


def validate_detector_rgb(raw):
    a = np.asarray(raw)
    if (a.shape != (3, 256, 256) or a.dtype != np.float32
            or not np.isfinite(a).all() or a.min() < 0 or a.max() > 255):
        raise ValueError('Exact inherited CHW float32 raw255 input is required')
    return a


def normalize_prompt(text):
    # Official model-card convention: lowercase class phrases, period separated.
    phrases = [p.strip().lower() for p in text.split('.') if p.strip()]
    if not phrases:
        raise ValueError('At least one explicit task/robot protection phrase is required')
    return '. '.join(phrases) + '.'


def detector_resize_shape(height, width, shortest_edge=800, longest_edge=1333):
    if not (32 <= height <= 4096 and 32 <= width <= 4096 and shortest_edge in (480,800)):
        raise ValueError('Declared supported source dimensions/resolution required')
    scale = min(shortest_edge/min(height,width), longest_edge/max(height,width))
    return max(1,int(height*scale)), max(1,int(width*scale))


def proposal_foreground(raw, detections, producer, required_labels, policy=BoxPolicy()):
    """Never promotes automatic detections to a reviewed training cache."""
    validate_detector_rgb(raw)
    for d in detections:
        if not (np.isfinite(d['score']) and 0<=d['score']<=1 and isinstance(d['label'],str)):
            raise ValueError('Invalid detection confidence/label')
        expanded_box_mask([d['box']],policy=policy)
    selected = [d for d in detections if d['score'] >= policy.score_threshold]
    mask, expanded = expanded_box_mask([d['box'] for d in selected], policy=policy)
    sha = hashlib.sha256(json.dumps(producer, sort_keys=True).encode()).hexdigest()
    foreground = Foreground(mask, array_sha(raw), sha, False, tuple(required_labels), ())
    return foreground, expanded


class GroundingBoxDetector:
    """Frozen public HF implementation; strict local files, no remote code.

    Its ImageNet normalization is for the OFFLINE detector only. Student/Qwen,
    action norm, weak teacher images and their normalizations are not changed.
    """
    def __init__(self, model_dir, *, device='cuda:0', shortest_edge=800,
                 policy=BoxPolicy(), expected_files=None):
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        path = Path(model_dir)
        if expected_files is None or 'model.safetensors' not in expected_files:
            raise ValueError('Pinned model file SHA inventory is required')
        for name, info in expected_files.items():
            p = (path / name).resolve()
            if not p.is_relative_to(path.resolve()) or file_sha(p) != info['sha256']:
                raise ValueError('Public model identity changed: ' + name)
        if shortest_edge not in (480, 800):
            raise ValueError('Only explicit measured detector resolutions are supported')
        self.torch, self.device, self.policy = torch, device, policy
        self.shortest_edge = shortest_edge
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            path, local_files_only=True, use_safetensors=True).to(device).eval()
        self.model.requires_grad_(False)
        self.provenance = dict(name='IDEA-Research/grounding-dino-tiny',
            implementation='transformers', transformers_version=transformers.__version__,
            torch_version=torch.__version__, weights_sha256=expected_files['model.safetensors']['sha256'],
            thresholds=asdict(policy), shortest_edge=shortest_edge,
            normalization='raw255 -> HF rescale once /255 -> ImageNet mean/std; canonical CHW preserved',
            detector_resize='explicit torchvision bilinear antialias float32; HF do_resize=False, no PIL/unit guessing',
            input_rgb='exact inherited decode_rgb + torchvision Resize256 antialias float32',
            review_protocol='automatic proposal only; independent visible-instance review required')

    def _sync(self):
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def _prepare_inputs(self, images, prompts):
        from torchvision.transforms.v2.functional import resize
        from torchvision.transforms import InterpolationMode
        resized=[]
        for image in images:
            h,w=image.shape[1:]
            output_shape=detector_resize_shape(h,w,self.shortest_edge)
            # HF 4.57 slow/PIL resize rejects noninteger raw255 and guesses dark
            # float images as unit-scaled. Keep declared float values throughout.
            t=self.torch.from_numpy(np.ascontiguousarray(image))
            resized.append(resize(t,list(output_shape),interpolation=InterpolationMode.BILINEAR,
                                  antialias=True).numpy())
        result=self.processor(images=resized,text=prompts,return_tensors='pt',padding=True,
            input_data_format='channels_first',do_resize=False,do_rescale=True,do_normalize=True)
        if result['input_ids'].shape[-1]>self.model.config.max_text_len:
            raise ValueError('Task prompt exceeds detector text budget; no silent category truncation')
        return result

    def predict_batch(self, raws, prompts, *, minimum_score=.12, detector_images=None):
        if len(raws) != len(prompts) or not raws or not 0 < minimum_score <= self.policy.score_threshold:
            raise ValueError('Aligned nonempty inputs and a lower/equal proposal threshold required')
        for raw in raws:
            validate_detector_rgb(raw)
        prompts = [normalize_prompt(t) for t in prompts]
        if detector_images is not None:
            from torchvision.transforms.v2.functional import resize
            if len(detector_images)!=len(raws):raise ValueError('Source/canonical image count mismatch')
            for original,canonical in zip(detector_images,raws):
                if (original.ndim!=3 or original.shape[0]!=3 or original.dtype!=np.float32
                        or not np.isfinite(original).all() or original.min()<0 or original.max()>255):
                    raise ValueError('Original source must be finite float32 CHW raw255')
                check=resize(self.torch.from_numpy(np.ascontiguousarray(original)),[256,256],antialias=True).numpy()
                if not np.array_equal(check,canonical):
                    raise ValueError('Original detector RGB does not reconstruct canonical training RGB')
        self._sync()
        start = time.perf_counter()
        # Keep float32 CHW raw255; do not round canonical pixels to uint8/PIL.
        inputs = self._prepare_inputs(raws if detector_images is None else detector_images,prompts)
        prepared = time.perf_counter()
        inputs = inputs.to(self.device)
        self._sync()
        transferred = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        self._sync()
        inferred = time.perf_counter()
        results = self.processor.post_process_grounded_object_detection(outputs,
            inputs.input_ids, threshold=minimum_score, text_threshold=self.policy.text_threshold,
            target_sizes=[(256, 256)] * len(raws))
        detections = []
        for r in results:
            labels = r['text_labels'] if 'text_labels' in r else r['labels']
            ds = [dict(box=[float(v) for v in b], score=float(s), label=str(t))
                  for b, s, t in zip(r['boxes'].cpu().tolist(), r['scores'].cpu().tolist(), labels)]
            if any(not np.isfinite(d['score']) for d in ds):
                raise ValueError('Nonfinite detector score')
            expanded_box_mask([d['box'] for d in ds], policy=self.policy)
            detections.append(ds)
        self._sync()
        end = time.perf_counter()
        return detections, dict(images=len(raws), preprocess_s=prepared-start,
            host_to_device_s=transferred-prepared, forward_s=inferred-transferred,
            postprocess_s=end-inferred, detection_wall_s=end-start,
            input_shape=list(inputs['pixel_values'].shape), minimum_score=minimum_score)
