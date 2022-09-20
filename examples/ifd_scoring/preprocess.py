import os
import re
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import contextlib
import copy
from multiprocessing import Process, Queue, Pool
import glob
import json
import lmdb

def write_lmdb(data_path='./', save_path='./', prefix='competition_'):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path,'test.csv'))
    train_systems = train['System_ID'].unique()
    test_systems = test['System_ID'].unique()

    for name, systems in [('train',train_systems),
                    ('test', test_systems)][1:]:
        outputfilename = os.path.join(save_path, prefix + name + '.lmdb')
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        i = 0
        for system_name in systems:
            d = json.load(open(os.path.join(data_path, 'conformers', system_name, 'coords.json'),'r'))
            for confid in d.keys():
                dd = d[confid]
                atoms = dd['ligand_atoms']
                coords = np.array(dd['ligand_coords'], dtype=np.float32)
                patoms = dd['pocket_atoms']
                pcoords = np.array(dd['pocket_coords'], dtype=np.float32)
                if name == 'train':
                    rmsd = train[(train['System_ID']==system_name) & (train['Complex_ID']==confid)]['RMSD_Ligand'].values[0]
                    chi = train[(train['System_ID']==system_name) & (train['Complex_ID']==confid)]['Delta_Chi_1'].values[0]
                    rmsd_side = train[(train['System_ID']==system_name) & (train['Complex_ID']==confid)]['RMSD_Pocket_Sidechain'].values[0]
                else:
                    # rmsd = test[(test['System_ID']==system_name) & (test['Complex_ID']==confid)]['RMSD_Ligand'].values[0]
                    # chi = test[(test['System_ID']==system_name) & (test['Complex_ID']==confid)]['Delta_Chi_1'].values[0]
                    # rmsd_side = test[(test['System_ID']==system_name) & (test['Complex_ID']==confid)]['RMSD_Pocket_Sidechain'].values[0]      
                    rmsd = chi = rmsd_side = 0.0

                if len(atoms)>0 and len(patoms)>0:
                    inner_output = pickle.dumps({
                                                'atoms':atoms, 
                                                'coordinates':coords, 
                                                'pocket_atoms':patoms, 
                                                'pocket_coordinates':pcoords,
                                                'label': [rmsd,chi,rmsd_side],
                                                'pocket':system_name, 
                                                'scaffold':system_name,
                                                'confid':confid})
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 1000 == 0:
                        print(i)
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)

        print('{} process {} lines'.format(name, i))
        txn_write.commit()
        env_new.close()


if __name__ == '__main__':
    write_lmdb()
