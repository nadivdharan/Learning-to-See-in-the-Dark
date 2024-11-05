import argparse
import glob
import os
import tensorflow as tf
from tqdm import tqdm
import rawpy
import numpy as np
from pathlib import Path


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


input_dir = '/data/data/datasets/SID/Sony/short/'
gt_dir = '/data/data/datasets/SID/Sony/long/'
split_map = {"train": "0",
             "val": "2",
             "test": "1"}
default_split = "val"

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', help='model config path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-images', type=int, default=64, help='Number of images for calibration set')
    parser.add_argument('--out', type=str, default=None, help='path to save .npy calibration set')
    parser.add_argument('--split', type=str, default='val', help='Split to take images from', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    return args

def run(args):
    num_images = args.num_images
    split = args.split
    # checkpoint_dir = args.ckpt_dir
    
    # get test IDs
    test_fns = glob.glob(gt_dir + f'/{split_map[split]}*.ARW')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
    
    # tf.compat.v1.disable_v2_behavior()
    # sess = tf.compat.v1.Session()
    # in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
    # gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
    # out_image = network(in_image)
    
    # saver = tf.compat.v1.train.Saver()
    # sess.run(tf.compat.v1.global_variables_initializer())
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # if ckpt:
    #     print('loaded ' + ckpt.model_checkpoint_path)
    #     saver.restore(sess, ckpt.model_checkpoint_path)
        
    count = 0
    # for test_id in tqdm(test_ids, total=num_images):
    # for test_id in tqdm(test_ids, total=len(test_ids)):
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        # in_files = glob.glob(input_dir + '%05d_*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            print(in_fn, f' {count+1}/{num_images}')
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
            
            input_full = np.minimum(input_full, 1.0)
            if count == 0:
                shape = input_full.shape[1:]
                calib_set = np.zeros((num_images, *shape))
            calib_set[count] = input_full
            
            count += 1
            if count == num_images:
                break
        if count == num_images:
            break

    out_path = Path(os.getcwd()) / 'calib_set.npy'  if args.out is None else args.out
    np.save(f'{out_path}', calib_set)
    print(f'Calibration set saved at {out_path}')             

if __name__ == '__main__':
    args = parse_args()
    run(args)
