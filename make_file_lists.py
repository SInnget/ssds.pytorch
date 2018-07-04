import os
import os.path
import glob
import argparse
import sys
from lib.utils.config_parse import cfg_from_file

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Make training data list')
    parser.add_argument('--cfg', dest='config_file',
                        help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


args = parse_args()
if args.config_file is not None:
    cfg_from_file(args.config_file)

from lib.utils.config_parse import cfg

cwd = cfg.DATASET.DATASET_DIR
# print(cwd)
# print(os.listdir(cwd))
txt_path = os.path.join(cwd, 'ImageSets/Main')
train_txt = txt_path + '/' + cfg.DATASET.TRAIN_SETS[0][1] + '.txt'
# print(train_txt)
# file = open(train_txt, 'w')
with open(train_txt, 'w') as file:
    anno_path = os.path.join(cwd, 'Annotations')
    image_path = os.path.join(cwd, 'JPEGImages')
    search_files = anno_path + '/*.xml'
    for name in glob.glob(search_files):
        print('name:', name)
        tmp = name.split('/')[-1]
        tmp = tmp.split('.')[0]
        image = image_path + '/' + tmp + '.jpg'
        if os.path.isfile(image):
            print(image)

            file.writelines(tmp+'\n')
