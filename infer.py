import os
import glob

import imageio
import visdom
import numpy as np

import torch
from torch.autograd import Variable

from tqdm import tqdm
from scipy.misc import imread, imsave
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk, label
from skimage.segmentation import find_boundaries

from inputs import get_watermark_template, remove_watermark
from unet import UNetVanilla

vis = visdom.Visdom()


def blend(image, label, curve_mask=None, alpha=0.3):
    """"Simulate colormap `jet`."""
    image, label = [item.astype(float) for item in [image, label]]
    r = label * alpha + 20
    b = (image + image.mean()) * (1 + alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.3)
    if curve_mask is not None:
        # curve_mask = curve_mask[..., None]  # for broadcast
        # rgb += curve_mask * 0.5
        rgb[..., 1] += curve_mask * 0.5  # add to blue channel
    # vis.image(rgb.transpose(2, 0, 1))
    return rgb.astype(np.uint8)


def loop_close(num, bw, radius=35):
    while num > 1:
        bw = binary_closing(bw, disk(radius))
        _, num = label(bw, return_num=True)
        radius += 10
        # vis.image(bw.astype(np.uint8) * 255)
    return bw


def loop_corner(coords, bw, w, rw, radius=35):
    """
    Argument:
        coords: coordinates of detected corners
        bw: black-and-white label
        w:  width of label
        rw: region of interest
        radius: element radius to closing the gap between corners
    """
    while len(coords):
        coords = corner_peaks(corner_harris(bw[:, w // 2 - rw: w // 2 + rw], k=0.2), min_distance=10)
        bw = binary_closing(bw, disk(radius))
        radius += 10
    return bw


def remove_isolation(mask, roi_width=30):
    """Remove discrete pixels and close gaps
    Argument:
        roi width: corner detection range
    """
    # 1) Filling holes
    bw = label(mask == 1)
    bw = remove_small_objects(bw, min_size=4096, connectivity=2)
    bw = remove_small_holes(bw, min_size=4096, connectivity=2)
    # 2) Detect evil interval
    _, num = label(bw, return_num=True)
    if num > 1:
        print('Fixed discontinuity')
        bw = loop_close(num, bw)
    # 3) Detect breach in the edge
    w, rw = mask.shape[1], roi_width
    coords = corner_peaks(corner_harris(bw[:, w // 2 - rw: w // 2 + rw], k=0.2), min_distance=10)
    if len(coords):
        print('Detected corner and filling')
        bw = loop_corner(coords, bw, w, rw)
    return bw.astype(np.uint8)


def fitting_curve(mask, margin=(60, 60)):
    """Compute thickness by fitting the curve
    Argument:
        margin: indicate valid mask region in case overfit

    Return:
        thickness: between upper and lower limbus
        curve_mask: the same shape as mask while labeled by 1
    """
    # 1. Find boundary
    bound = find_boundaries(mask, mode='outer')
    # 2. Crop marginal parts (may be noise)
    lhs, rhs = margin
    bound[:, :lhs] = 0  # left hand side
    bound[:, -rhs:] = 0  # right hand side
    # 3. Process upper and lower boundary respectively
    labeled_bound = label(bound, connectivity=bound.ndim)
    upper, lower = labeled_bound == 1, labeled_bound == 2
    # 1) fit poly
    f_up, f_lw = [np.poly1d(np.polyfit(np.where(limit)[1], np.where(limit)[0], 6)) for limit in [upper, lower]]
    # 2) interpolation
    width = mask.shape[1]
    x_cord = range(width)
    y_up_fit, y_lw_fit = [f(x_cord) for f in [f_up, f_lw]]

    thickness = (y_up_fit - y_lw_fit)[width // 2 - 10: width // 2 + 10]

    curve_mask = np.zeros_like(mask)
    y_up_fit, y_lw_fit = [np.array(y, dtype=int) for y in [y_up_fit, y_lw_fit]]  # int for slice
    curve_mask[y_up_fit[lhs: -rhs], x_cord[lhs: -rhs]] = 255
    curve_mask[y_lw_fit[lhs: -rhs], x_cord[lhs: -rhs]] = 255

    return abs(thickness.mean()), curve_mask


def name2digit(name, char='_'):
    return int(name.split(char)[-1][:-4])


def infer(model, model_name, infer_index=[1, 2, 3], dewatermark=True):
    model = model.cuda().eval()

    best_path = 'checkpoints/{}/{}_best.pth'.format(model_name, model_name)
    best_model = torch.load(best_path)
    print('===> Loading model from {}...'.format(best_path))
    model.load_state_dict(best_model)

    template_bw, surround_bw = get_watermark_template(1)  # utilize standard watermark

    for idx in infer_index:
        infer_path = sorted(glob.glob('data/frames_{}/*'.format(idx)), key=name2digit)
        print('===> Processing dataset: {}...'.format(idx))

        thicks = []
        for i, p in tqdm(enumerate(infer_path, start=1)):
            # Read frame and previous frame simultaneously
            if dewatermark:
                im = remove_watermark(imread(p), template_bw, surround_bw)
            else:
                im = imread(p)
            x = Variable(torch.from_numpy(im[None, None, ...]), volatile=True).float().cuda()

            if i == 1:
                lb = imread('data/label_init_{}.jpg'.format(idx))
                lb = (lb > lb.max() * 0.5).astype(float)
                y_prev = Variable(torch.from_numpy(lb[None, None, ...]), volatile=True).float().cuda()

            output = model(x, y_prev)
            if isinstance(output, (tuple, list)):
                output = output[0]

            pred = output.data.max(1)[1]
            y_prev = Variable(pred[None, ...], volatile=True).float()  # as next frame's input

            pred = pred[0].cpu().numpy()
            # Post-process
            display = remove_isolation(pred) * 255
            try:
                thickness, curve_mask = fitting_curve(display)
                thicks.append(thickness)
            except IndexError:
                curve_mask = None
                thicks.append(0)
                print('Oops, fail to detect {}th frame...'.format(i))

            # Save and Display
            deploy_dir = ['deploy/{}/{}_{}'.format(model_name, subdir, idx) for subdir in ['bw', 'blend']]
            [os.makedirs(dd) for dd in deploy_dir if not os.path.exists(dd)]

            imsave(deploy_dir[0] + '/{}.jpg'.format(i), display)
            imsave(deploy_dir[1] + '/{}.jpg'.format(i), blend(im, display, curve_mask))
            # vis.image(im, opts=dict(title='image: {}'.format(i)))
            # vis.image(display, opts=dict(title='image: {}'.format(i)))

        np.save('deploy/{}/thickness_{}.npy'.format(model_name, idx), thicks)


def generate_gif(model_name, infer_index):
    for idx in infer_index:
        print('===> Generating Gif: {}...'.format(idx))
        with imageio.get_writer('deploy/{}/blend_{}.gif'.format(model_name, idx), mode='I') as writer:
            for im_path in sorted(glob.glob('deploy/{}/blend_{}/*.jpg'.format(model_name, idx)),
                                  key=lambda k: name2digit(k, '/')):
                image = imread(im_path)
                writer.append_data(image)


if __name__ == '__main__':
    torch.cuda.set_device(1)

    model = UNetVanilla()
    model_name = 'UNetVanilla'
    infer(model, model_name, infer_index=[1, 2, 3, 4, 5], dewatermark=True)
    generate_gif(model_name, infer_index=[1, 2, 3, 4, 5])
