from __future__ import print_function

import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
# from lib.ssds_train import train_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a ssds.pytorch network')
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
from lib.modeling.model_builder import create_model

# print(cfg.MODEL.SSDS)
# load the weights of original model
print(cfg.WEIGHT_CONVERT.ORIGINAL_WEIGHT)
weight_to_be_modified = torch.load(
    cfg.WEIGHT_CONVERT.ORIGINAL_WEIGHT, map_location='cpu')
print('Successfully load the weight of original model!')
# print(weight_to_be_modified)
model, _ = create_model(cfg.MODEL)
print('The new model has been loaded.')

components_to_be_changed = [
    t.strip() for t in cfg.WEIGHT_CONVERT.COMPS_TO_CONVERT.split(',')]

weights_dict = model.state_dict()
weight_keys = model.state_dict().keys()

for key in weight_keys:
    # print(key)
    keys = key.split('.')
    # if (set(key.split('.') & set(components_to_be_changed))): # check whether pref in the compo
    if keys[0] in components_to_be_changed:
        if 'weight' in keys:
            weight_to_be_modified[key] = nn.init.xavier_normal_(torch.zeros_like(weights_dict[key]))
            print(weight_to_be_modified[key].size())
            # print(weight_to_be_modified[key].device)
        if 'bias' in keys:
            weight_to_be_modified[key] = nn.init.constant_(
                torch.zeros_like(weights_dict[key]), 0.01)
            print(weight_to_be_modified[key].size())
            # print(weight_to_be_modified[key].device)
torch.save(weight_to_be_modified, cfg.RESUME_CHECKPOINT)
print('The modified weight has been successfully saved at {}'.format(
    os.getcwd() + cfg.RESUME_CHECKPOINT.strip('.')))

# print(weight_to_be_modified)
# print(model)

# print(new_model.loc)
# print(new_model.conf)




# print(priorbox)
