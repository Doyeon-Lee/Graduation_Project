from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('..\\..\\FairMOT\\src')

import _init_paths

import logging
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else f'../output/video/{opt.input_video_name}'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    if opt.input_format == 'photo':
        dataloader = datasets.LoadImages(opt.input_video, opt.img_size)
        frame_rate = 1
    else:
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
        frame_rate = dataloader.frame_rate
    result_filename = os.path.join(result_root, f'results{opt.input_video_name}_0.csv')

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, f'results{opt.input_video_name}_0.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)

    return result_filename


def tracking(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init(args)
    return demo(opt)
