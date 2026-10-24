"""Explicit adapter for the pinned augmentation's float RGB unit mismatch.

The raw HDF5 reader and deploy wrapper both use float RGB in [0,255].
The pinned color augmentation returns float RGB in [0,1], while its downstream
processor and teacher still divide by 255. Keep that boundary in raw RGB units.
Install only in a new training process; never patch a historical run in place.
"""


def install_raw255_augmentation(transform_module):
    import torch
    if hasattr(transform_module, '_raw255_augmentation_original'):
        return False
    original = transform_module.apply_visual_augmentation

    def apply_visual_augmentation_raw255(visual, params):
        if visual.dtype == torch.uint8:
            return original(visual, params)
        if not visual.is_floating_point():
            raise TypeError('Raw RGB must be uint8 or floating point in [0,255]')
        if not torch.isfinite(visual).all() or visual.min() < 0 or visual.max() > 255:
            raise ValueError('Raw RGB is outside the declared [0,255] contract')
        # Normalize explicitly even for nearly black images with max <= 1.
        # The original operation/color parameters are retained, without an
        # integer conversion or an additional sampling operation.
        unit = visual / 255.0
        augmented = original(unit, params)
        return (augmented * 255.0).to(dtype=visual.dtype)

    transform_module._raw255_augmentation_original = original
    transform_module.apply_visual_augmentation = apply_visual_augmentation_raw255
    return True
