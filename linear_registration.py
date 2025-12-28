import SimpleITK as sitk
from pathlib import Path
import numpy as np
from utils import *

def register_linear(fix_im, mov_im, fix_mask=None, mov_mask=None, verbose=False):
    fix_need_lateral_crop = False
    mov_need_lateral_crop = False
    # If no mask is provided for either of the image, create one via HU threshold and perform basic cleaning to remove
    # some irrelevant highly attenuating elements.
    if fix_mask is None:
        if verbose: print('Creating bone mask for fixed image...')
        fix_mask = mask_from_hu(fix_im, hu_min=185, hu_max=500, closing_kernel_size=10, verbose=verbose)
        fix_mask = clean_bone_mask(fix_mask)
        fix_need_lateral_crop = True
    if mov_mask is None:
        if verbose: print('Creating bone mask for moving image...')
        mov_mask = mask_from_hu(mov_im, hu_min=185, hu_max=500, closing_kernel_size=10, verbose=verbose)
        mov_mask = clean_bone_mask(mov_mask)
        mov_need_lateral_crop = True

    # Crop fixed and moving images to ROI around right pelvis and femur
    if verbose: print('Cropping fixed image to ROI...')
    fix_im_roi, fix_mask_roi = crop_roi_from_mask_multi(fix_im, fix_mask)
    if fix_need_lateral_crop:
        fix_im_roi = crop_lateral(fix_im_roi, 'r')
        fix_mask_roi = crop_lateral(fix_mask_roi, 'r')
    if verbose: print('Cropping moving image to ROI...')
    mov_im_roi, mov_mask_roi = crop_roi_from_mask_multi(mov_im, mov_mask)
    if mov_need_lateral_crop:
        mov_im_roi = crop_lateral(mov_im_roi, 'r')
        # mov_mask_roi = crop_lateral(mov_mask_roi, 'r')

    if verbose: print('Estimating affine transformation...')
    lin_tfm = est_lin_transf(fix_im_roi, mov_im_roi, mask_ref=fix_mask_roi, verbose=verbose)
    if verbose: print('Applying affine transformation...')
    mov_im_reg = apply_lin_transf(mov_im, lin_tfm, fix_im)
    mov_mask_reg = apply_lin_transf(mov_mask, lin_tfm, fix_im)
    if verbose: print('Linear registration finished.')

    if fix_need_lateral_crop:
        return {
            'registered_image': mov_im_reg,
            'registered_mask': mov_mask_reg,
            'affine_tfm': lin_tfm,
            'fixed_mask': fix_mask,
        }
    return {
        'registered_image': mov_im_reg,
        'registered_mask': mov_mask_reg,
        'affine_tfm': lin_tfm,
    }


