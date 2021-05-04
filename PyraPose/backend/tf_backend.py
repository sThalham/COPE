import tensorflow


def ones(*args, **kwargs):
    return tensorflow.ones(*args, **kwargs)


def transpose(*args, **kwargs):
    return tensorflow.transpose(*args, **kwargs)


def map_fn(*args, **kwargs):
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.image.resize(images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tensorflow.where(*args, **kwargs)
