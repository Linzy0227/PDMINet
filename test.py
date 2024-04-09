import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn

from modules import fusionseg

cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_softmax


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='', type=str)

parser.add_argument('--project_root', default='', type=str)

parser.add_argument('--root', default='', type=str)

parser.add_argument('--val_file', default='', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--epoch', default='', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='Mymodal', type=str)

parser.add_argument('--num_cls', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = fusionseg.Model(num_cls=args.num_cls)

    model = torch.nn.DataParallel(model).cuda()

    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        # args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(args.resume))
    else:
        print('There is no resume file to load!')

    val_set = BraTS(args.val_file, args.root, mode='test')

    print('Samples for valid = {}'.format(len(val_set)))

    valid_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(args.project_root, args.output_dir,args.epoch, args.submission)
    visual = os.path.join(args.project_root, args.output_dir, args.epoch, args.visual)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=args.resume,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=val_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True,
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(val_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()



