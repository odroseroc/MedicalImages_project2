import SimpleITK as sitk

def mask_from_hu(img, hu_min=206, hu_max=None, closing_kernel_size=4, verbose=False):
    """
    Create a simple mask from HU thresholding.
    img: SimpleITK image (CT en HU)
    hu_min: minimum threshold for target structures
    returns: SimpleITK binary mask (UInt8)
    """
    if verbose:
        print(f'Generating mask from HU thresholding with HU_min={hu_min}, HU_max={hu_max}...')
    mask = img >= hu_min
    if hu_max is not None:
        mask &= img < hu_max
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    # mask = sitk.BinaryMorphologicalOpening(mask, [2,2,2])
    mask = sitk.BinaryMorphologicalClosing(mask, [closing_kernel_size,closing_kernel_size,closing_kernel_size])
    mask = sitk.BinaryFillhole(mask)
    if verbose:
        print(f'Mask generated!')
    return mask

def clean_bone_mask(mask, size_threshold=100000):
    """
    Clean the mask based on size threshold. This aims at eliminating elements of the medical table that were included in
    the HU thresholding.
    :param mask: Mask obtained from HU thresholding.
    :param size_threshold: Size of the components to be removed.
    :return: SimpleITK binary mask (UInt8): cleaned mask.
    """
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    clean = sitk.Image(cc.GetSize(), sitk.sitkUInt8)
    clean.CopyInformation(cc)
    for l in stats.GetLabels():
        if stats.GetPhysicalSize(l) > size_threshold:  # ajusta umbral
            clean = clean | sitk.Equal(cc, l)
    return clean