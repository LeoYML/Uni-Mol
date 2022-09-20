import pandas as pd
import numpy as np

data = pd.read_pickle('./IFD_run_lr_3e-4_bs_32_epoch_10_wp_0.06_fold_0_test.out.pkl')
system_list, conformer_list = [], []
rmsd_predict, chi_predict = [], []
for chunk in data:
    system_list.extend(chunk['system'])
    conformer_list.extend(chunk['confid'])
    rmsd_predict.extend(chunk['rmsd_predict'].view(-1).cpu().numpy().tolist())
    chi_predict.extend(chunk['chi_predict'].view(-1).cpu().numpy().tolist())

sub = pd.DataFrame({'System_ID':system_list,
                    'Complex_ID':conformer_list,
                    'Ligand_score':rmsd_predict,
                    'Delta_Chi_1_score':chi_predict})


sample = pd.read_csv('./sample.submission.csv')

sub = pd.merge(sample[['System_ID', 'Complex_ID']], sub, on=['System_ID', 'Complex_ID'], how='left').fillna(100.0)

sub.to_csv('unimol.baseline.csv', index=False, header=True)