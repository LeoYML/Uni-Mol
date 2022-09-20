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
import warnings
warnings.filterwarnings(action='ignore')

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
        rmsd_output, chi_output = net_output

        rmsd_target = sample["target"]["all_target"][:,0]
        chi_target = sample["target"]["all_target"][:,1]
        rmsd_loss = self.compute_rmsd_loss(rmsd_output, rmsd_target)
        chi_loss = self.compute_rmsd_loss(chi_output, chi_target)

        loss = rmsd_loss + chi_loss
        sz = sample["target"]["all_target"].size(0)
        logging_output = {
            "loss": loss.data,
            "rmsd_loss": rmsd_loss.data,
            "chi_loss": chi_loss.data,
            "bsz": sz,
            "sample_size": sz,
        }
        if not self.training:
            logging_output['system'] = sample["system"]
            logging_output['confid'] = sample["confid"]
            logging_output['rmsd_target'] = rmsd_target
            logging_output['chi_target'] = chi_target
            logging_output['rmsd_predict'] = rmsd_output
            logging_output['chi_predict'] = chi_output

        return loss, sz, logging_output

    def compute_rmsd_loss(self, rmsd_output, rmsd_target):
        normal_rmsd_target = torch.log(rmsd_target + 1.0).float()
        mse_loss = F.l1_loss(rmsd_output.float(), normal_rmsd_target, reduce='none')
        mask = rmsd_target < 2.5
        weight = mask.type_as(rmsd_output) * 3.0 + 1.0
        return (mse_loss * weight).sum()

    def compute_chi_loss(self, chi_output, chi_target):
        normal_chi_target = torch.log(chi_target / 15.0 + 1.0).float()
        mse_loss = F.mse_loss(chi_output.float(), normal_chi_target, reduce='none')
        mask = chi_target < 45.0
        weight = mask.type_as(chi_output) * 3.0 + 1.0
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
        chi_loss = sum(log.get("chi_loss", 0) for log in logging_outputs)
        if chi_loss > 0:
            metrics.log_scalar(
                "chi_loss", chi_loss / sample_size, sample_size, round=5
            )
        
        if split in ['valid', 'test']:
            system = [item for log in logging_outputs for item in log.get("system")]
            rmsd_target = torch.cat([log.get("rmsd_target", 0) for log in logging_outputs], dim=0)
            rmsd_predict = torch.cat([log.get("rmsd_predict", 0) for log in logging_outputs], dim=0)
            df = pd.DataFrame({'RMSD_Ligand':rmsd_target.view(-1).cpu().numpy(),"Ligand_score":rmsd_predict.view(-1).cpu().numpy(),"System_ID":system})
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
