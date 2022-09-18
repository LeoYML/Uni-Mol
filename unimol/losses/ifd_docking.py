# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import roc_auc_score
from collections.abc import Iterable

@register_loss("ifd_scoring")
class IFDScoringLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        rmsd_output = net_output[:,0]
        rmsd_sidechain_output = net_output[:,1]

        rmsd_target = sample["target"]["all_target"][:,0]
        rmsd_loss = self.compute_loss(rmsd_output, rmsd_target)

        loss = rmsd_loss
        logging_output = {
            "loss": loss.data,
            "rmsd_loss": rmsd_loss.data,
            "bsz": sample["target"]["all_target"].size(0),
            "sample_size": sample["target"]["all_target"].size(0),
        }
        if not self.training:
            logging_output['system'] = sample["system"]
            logging_output['target'] = rmsd_target
            logging_output['predict'] = rmsd_output
            logging_output['confid'] = sample["confid"]

        return loss, 1, logging_output

    def compute_loss(self, rmsd_output, rmsd_target):
        mse_loss = F.l1_loss(rmsd_output, rmsd_target, reduce='none')
        mask = rmsd_target < 2.5
        weight = mask.type_as(rmsd_output) * 5.0 + 1.0

        return (mse_loss * weight).sum()
        
    
    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=5
        )
        metrics.log_scalar(
            "{}_loss".format(split), loss_sum / sample_size, sample_size, round=5
        )
        rmsd_loss = sum(log.get("rmsd_loss", 0) for log in logging_outputs)
        if rmsd_loss > 0:
            metrics.log_scalar(
                "rmsd_loss", rmsd_loss / sample_size, sample_size, round=5
            )
        
        if split in ['valid','test']:
            system = [item for log in logging_outputs for item in log.get("system")]
            target = torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
            predict = torch.cat([log.get("predict", 0) for log in logging_outputs], dim=0)
            df = pd.DataFrame({'RMSD_Ligand':target.cpu(),"Ligand_score":predict.cpu(),"System_ID":system})
            results = calc_metrics(df) 
            top1_sr, top2_sr, top5_sr = results['top1_sr'].mean(), results['top2_sr'].mean(),results['top5_sr'].mean()
            metrics.log_scalar(
                "{}_top1_sr".format(split), top1_sr, sample_size, round=5
            )
            metrics.log_scalar(
                "{}_top2_sr".format(split), top2_sr, sample_size, round=5
            )
            metrics.log_scalar(
                "{}_top5_sr".format(split), top5_sr, sample_size, round=5
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def single_score(tmp):
    stats = pd.Series()
    tmp['Ligand_score_rank'] = tmp['Ligand_score'].rank(method='first')
    ligand_score = []
    for _rank in [1,2,5]:
        for _col in ['Ligand_2.5']:
            _score = tmp[tmp['Ligand_score_rank']<=_rank][_col]
            _score = _score.mean()
            ligand_score.append(int(_score>0))
    stats['top1_sr'] = ligand_score[0]   
    stats['top2_sr'] = ligand_score[0]
    stats['top5_sr'] = ligand_score[1]
    return stats

def calc_metrics(df):

    df['Ligand_2.5'] = (df['RMSD_Ligand']<=2.5).astype(int)
    stats = df.groupby(['System_ID']).apply(single_score)
    
    return stats
