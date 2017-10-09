import time
import glob

import torch
import numpy as np

import torch.utils.data as data

from scipy.misc import imread
from collections import Counter
from skimage import img_as_float
from skimage.exposure import adjust_gamma
from skimage.transform import AffineTransform, PiecewiseAffineTransform, warp, resize
from skimage.morphology import binary_dilation, disk


def get_watermark_template(frame_index):
    """Eliminate the effect of different timestamp.
    Returns:
        template: binary image of watermark
        surround: dilated template for later computation of surrounding intensity
    """
    frames_path = glob.glob('data/frames_{}/*.jpg'.format(frame_index))
    template = np.zeros_like(imread(frames_path[0]), dtype=bool)
    for p in frames_path:
        im = imread(p)
        bw = im > im.max() * 0.5  # empirical threshold
        template |= bw
    template_bw = binary_dilation(template, selem=disk(1))
    surround_bw = binary_dilation(template_bw, selem=disk(5)) ^ template_bw
    return template_bw, surround_bw


def remove_watermark(frame, template_bw, surround_bw):
    """Remove watermark by fill surrounding intensity."""
    # Gray value around template
    surround_intensity = frame[surround_bw].mean()
    # Subtract
    demark = np.select([~template_bw], [frame], default=surround_intensity)
    return demark


def warp_pairs(image, label, tform):
    """Apply tform to
    image: filled by background intensity,
    label: filled with constant 0
    """
    bg = Counter(image.ravel()).most_common(1)[0][0]

    image = warp(image, tform, cval=bg, preserve_range=True)
    label = warp(label, tform, cval=0, preserve_range=True)
    label = label > 0.5 * label.max()
    return image, label.astype(float)


def generate_sine_frame(image_init, label_init,
                        height_limit=(0, 20),
                        scale_limit=(1.1, 1.3),
                        shift_limit=(25, 35)):
    """Piecewise Affine to simulate sinusoidal corneal limbus.

    Argument:
        image_init: 1st frame WITHOUT watermark
        label_init: 1st corresponding label
        height_limit: peak of sine
        scale_limit: < 1: enlarge; > 1: shrink
        shift_limit: horizontal shift, make the shrunken curve lower
    Returns:
        image: warped image (float, range(0, 255))
        label: warped label (float, set{0, 1})
    """
    height, width = image_init.shape
    src_x, src_y = np.linspace(0, width, 20), np.linspace(0, height, 10)
    src_x, src_y = np.meshgrid(src_x, src_y, indexing='ij')
    # Cartesian indexing, src.shape: (num of points, 2)
    src = np.vstack([src_x.flat, src_y.flat]).T
    # Add sinusoidal oscillation to y coordinates
    half_period_num = 3  # overall sinuous shape
    max_height = np.random.uniform(*height_limit)  # the max value of sine

    dst_x = src[:, 0]
    dst_y = src[:, 1] + np.sin(np.linspace(0, half_period_num * np.pi, src.shape[0])) * max_height

    scale = np.random.uniform(*scale_limit)
    shift = np.random.uniform(*shift_limit)
    dst_y = scale * dst_y - shift
    dst = np.vstack([dst_x, dst_y]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    return warp_pairs(image_init, label_init, tform)


def simulate_prev_label(label, scale_limit=(0.9, 1.1), trans_limit=(-0.025, 0.025)):
    """Simulate previous frame
    1. random affine transformation to present motion
    2. dilation to coarsen
    """

    trans_y = np.random.uniform(*np.multiply(label.shape, trans_limit))
    tform = AffineTransform(scale=scale_limit, translation=(0, trans_y))
    label = warp(label, tform, cval=0, preserve_range=True)
    bw = label > 0.5 * label.max()
    return binary_dilation(bw, selem=disk(1))


def random_gamma(image):
    """Adjust images' intensity"""

    # `inputs` has shape [height, width] with value in [0, 1].
    gamma = np.random.uniform(low=0.9, high=1.1)
    return adjust_gamma(image, gamma)


def random_hflip(image, label, u=0.5):
    if np.random.random() < u:
        image = np.fliplr(image)  # may be wrong, take a look on axis
        label = np.fliplr(label)
    return image, label


class CornealLimbusDataset(data.Dataset):
    def __init__(self, mode='train', image_size=(200, 576), train_idx=[1, 2], test_idx=[3]):
        """Assume dataset is in directory '.data/frames_X/*' and '.data/label_init_X.jpg'

        Argument:
            mode: 'train' / 'test'
            train_idx: index of training set
            test_idx: index of test set

        """
        super(CornealLimbusDataset, self).__init__()
        self.image_size = image_size
        self.template_bw, self.surround_bw = get_watermark_template(train_idx[0])
        # Train 1 , Validation 1 , Test 1 :P
        frame_init = 'data/frames_{}/video_{}_1.jpg'

        self.train_path = [frame_init.format(i, i) for i in train_idx]
        self.test_path = [frame_init.format(i, i) for i in test_idx]

        self.image_path = eval('self.{}_path'.format(mode))
        self.label_path = ['data/label_init_{}.jpg'.format(p.split('_')[2]) for p in self.image_path]

    def __getitem__(self, index):
        # Set random seed for random augment
        np.random.seed(int(time.time()))

        # Load gray image, `F` (32-bit floating point pixels)
        im, lb = [imread(p[index], mode='F') for p in [self.image_path, self.label_path]]
        if self.image_size != (200, 576):
            im, lb = [resize(item, (200, 576)) for item in [im, lb]]

        # Remove watermark
        im = remove_watermark(im, self.template_bw, self.surround_bw)

        # Generate sinusoidal frame (~ truly middle frame)
        im, lb = generate_sine_frame(im, lb)

        # Augment
        # 1). Intensity
        im = random_gamma(im)
        # 2). Random horizontal flip
        im, lb = random_hflip(im, lb)

        # One more label
        prev_lb = simulate_prev_label(lb)

        # Normalize
        im = img_as_float(im)
        im, lb, prev_lb = [item[np.newaxis, ...].astype(np.float32) for item in [im, lb, prev_lb]]

        # Convert to Tensor
        return [torch.from_numpy(item) for item in [im, lb, prev_lb]]

    def __len__(self):
        return len(self.image_path)


def _test_data_loader(dataset=CornealLimbusDataset()):
    training_data_loader = data.DataLoader(dataset=dataset, num_workers=4, batch_size=4, shuffle=True)
    im, lb, prev_lb = next(iter(training_data_loader))
    if isinstance(lb, torch.FloatTensor):
        print(im.size(), 'image')
        print(lb.size(), 'label')
        print(prev_lb.size(), 'prev_lb')

        vis.images(im.numpy(), opts=dict(title='image', caption='Shape: {}'.format(im.size())))
        vis.images(lb.numpy(), opts=dict(title='label', caption='Shape: {}'.format(lb.size())))
        vis.images(prev_lb.numpy(), opts=dict(title='previous label', caption='Shape: {}'.format(prev_lb.size())))

    else:
        print(im.size(), 'image')
        print(lb, 'image path')
        vis.images(im.numpy(), opts=dict(title='Random selected image', caption='Shape: {}'.format(im.size())))


if __name__ == '__main__':
    import visdom

    vis = visdom.Visdom()

    _test_data_loader()
