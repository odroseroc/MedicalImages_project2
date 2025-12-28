import SimpleITK as sitk

def crop_roi_from_mask(img, mask, pad=10):
    """
    Select as region of interest only the parts of an image present in a mask.
    :param img:
    :param mask:
    :param pad:
    :return:
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    bbox = stats.GetBoundingBox(1)  # x,y,z, sx,sy,sz
    x,y,z,sx,sy,sz = bbox
    start = [max(0, x-pad), max(0, y-pad), max(0, z-pad)]
    size = [
        min(img.GetSize()[0]-start[0], sx+2*pad),
        min(img.GetSize()[1]-start[1], sy+2*pad),
        min(img.GetSize()[2]-start[2], sz+2*pad)
    ]
    roi_img  = sitk.RegionOfInterest(img, size, start)
    roi_mask = sitk.RegionOfInterest(mask, size, start)
    return roi_img, roi_mask


def crop_roi_from_mask_multi(img, mask, labels=None, pad=10):
    """
    Select as region of interest the parts of an image present in one or more mask labels.

    :param img: SimpleITK Image to crop
    :param mask: SimpleITK mask image with integer labels
    :param labels: List of integer labels to include. If None, all labels > 0 are used.
    :param pad: Number of voxels to pad around the bounding box
    :return: roi_img, roi_mask
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    # Si no se especifican etiquetas, usamos todas excepto el fondo (0)
    if labels is None:
        labels = [lbl for lbl in stats.GetLabels() if lbl != 0]

    if not labels:
        raise ValueError("No valid labels found in the mask.")

    # Inicializamos las listas para calcular la caja que contenga todas las etiquetas
    xmins, ymins, zmins = [], [], []
    xmaxs, ymaxs, zmaxs = [], [], []

    for lbl in labels:
        if not stats.HasLabel(lbl):
            continue
        bbox = stats.GetBoundingBox(lbl)
        x, y, z, sx, sy, sz = bbox
        xmins.append(x)
        ymins.append(y)
        zmins.append(z)
        xmaxs.append(x + sx)
        ymaxs.append(y + sy)
        zmaxs.append(z + sz)

    if not xmins:
        raise ValueError("None of the requested labels exist in the mask.")

    # Calculamos start y size con padding y límites de la imagen
    start = [
        max(0, min(xmins) - pad),
        max(0, min(ymins) - pad),
        max(0, min(zmins) - pad)
    ]

    size = [
        min(img.GetSize()[0] - start[0], max(xmaxs) - min(xmins) + 2 * pad),
        min(img.GetSize()[1] - start[1], max(ymaxs) - min(ymins) + 2 * pad),
        min(img.GetSize()[2] - start[2], max(zmaxs) - min(zmins) + 2 * pad)
    ]

    # Convertimos a enteros para SimpleITK
    start = [int(s) for s in start]
    size = [int(s) for s in size]

    roi_img = sitk.RegionOfInterest(img, size, start)
    roi_mask = sitk.RegionOfInterest(mask, size, start)

    return roi_img, roi_mask

def crop_lateral(img, side='r', pad=10):
    """
    Select as region of interest only the right or left side of the body.
    :param img:
    :param mask:
    :param side: 'r' (right) or 'l' (left)
    :param pad:
    :return:
    """
    img_size = img.GetSize()
    size = [
        img_size[0]//2 + pad,
        img_size[1],
        img_size[2],
    ]
    match side:
        case 'r':
            start=[0,0,0]
        case 'l':
            start=[img_size[0]//2 - pad, 0, 0]
        case _:
            raise ValueError(f"Only \'r\' and \'l\' are acceptable sides.")
    roi_img = sitk.RegionOfInterest(img, size, start)
    return roi_img

def downsample(img, factor):
    """
    Downsample a SimpleITK image by a given factor.
    Preserves physical space (origin, direction).
    Works correctly even if img was cropped with RegionOfInterest.
    """

    in_spacing = img.GetSpacing()
    in_size = img.GetSize()

    out_spacing = [s * factor for s in in_spacing]
    out_size = [int(round(sz / factor)) for sz in in_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())  # identidad

    return resampler.Execute(img)
