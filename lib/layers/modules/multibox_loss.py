# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        # print('loc size',  loc_data.size())
        # print('conf size', conf_data.size())

        num = loc_data.size(0) # batch size
        # print('targets size: ', len(targets) )  = 32

        # print('targets size: ', targets[0].size() ) = 32*  3* 5
        # print('targets size: ', targets[0] )   3 ground truth per image

        priors = self.priors
        # print('priors: ',  priors)
        # print('priors size: ',  priors.size())  # anchor boxes: 2990
        # priors = priors[:loc_data.size(1), :]  2990x4
        num_priors = (priors.size(0))
        # print('num priors', num_priors) # 2990

        num_classes = self.num_classes # 21
        # print('num classes: ', num_classes)
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4) # 32 x 2990 x 4
        conf_t = torch.LongTensor(num, num_priors) # 32 * 2990
        for idx in range(num):
            truths = targets[idx][:,:-1].data
            # print(truths.size()) # varies
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        # print('loc_t size: ', loc_t.size())
        conf_t = Variable(conf_t,requires_grad=False)
        # print('conf_t size: ', conf_t.size())
        pos = conf_t > 0

        # print(pos)
        # print('pos size: ',  pos.size())
        # num_pos = pos.sum()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print('pos_idx: ',pos_idx.size())
        loc_p = loc_data[pos_idx].view(-1,4)
        # print('loc_p: ',loc_p.size())
        loc_t = loc_t[pos_idx].view(-1,4)
        # print('loc t :', loc_t.size())
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # print('batch conf: ', batch_conf.size())
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        # print('loss_c shape', loss_c.size())
        # Hard Negative Mining
        pos1 = pos.view(-1, 1)

        # print('pos shape', pos.size(), '\n')
        loss_c[pos1] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        # print('loss_c shape', loss_c.size())
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        # print('num pos: ', num_pos.size())
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # print('num neg: ', num_neg.size(), '\n')
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print('neg ', neg.size())
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        # print('N ',  N)
        # print('N type: ', N.dtype)
        # # loss_c.float()
        # print('loss_c type: ',  loss_c.dtype)
        loss_c/=N
        loss_l/=N
        return loss_l,loss_c
