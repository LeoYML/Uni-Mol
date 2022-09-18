# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections.abc import Iterable

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
    RawLabelDataset,
)
from unimol.data import (
    KeyDataset,
    DistanceDataset,
    EdgeTypeDataset,
    CroppingDataset,
    RemoveHydrogenDataset,
    RightPadDatasetCoord,
    LMDBDataset,
    NormalizeDockingPoseDataset,
    PrependAndAppend2DDataset,
    ConformerSampleDockingIFDDataset,
    FoldLMDBDataset,
    StackedLMDBDataset,

)
from unicore import checkpoint_utils
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

@register_task("ifd_scoring")
class IFDscoringIFD(DpaieTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--task-name",
            type=str,
            default='PDBbind_pose',
            help='downstream task name'
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--nfolds",
            default=5,
            type=int,
            help="cross validation split folds"
        )
        parser.add_argument(
            "--cv-seed",
            default=42,
            type=int,
            help="random seed used to do cross validation splits"
        )
        parser.add_argument(
            "--fold",
            default=0,
            type=int,
            help='local fold used as validation set, and other folds will be used as train set'
        )
        parser.add_argument(
            "--only-fc",
            action='store_true',
            help="freezen backbone layer",            
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )  
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )  
        
    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.init_data()
        
    def init_data(self):
        data_path = os.path.join(self.args.data, self.args.task_name + '_train.lmdb')
        raw_dataset = LMDBDataset(data_path)
        train_folds = []
        for _fold in range(self.args.nfolds):
            if _fold == 0:
                cache_fold_info = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
            if _fold == self.args.fold:
                self.valid_dataset = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
            if _fold != self.args.fold:
                train_folds.append(FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
        self.train_dataset = StackedLMDBDataset(train_folds)
    
        test_data_path = os.path.join(self.args.data, self.args.task_name + '_test.lmdb')
        self.test_dataset = LMDBDataset(test_data_path)
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'valid':
            dataset = self.valid_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            print(split)

        tgt_dataset = KeyDataset(dataset, 'label')
        system_dataset = KeyDataset(dataset, 'pocket')
        confid_dataset = KeyDataset(dataset, 'confid')

        dataset = ConformerSampleDockingIFDDataset(dataset, self.seed, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates')
        dataset = RemoveHydrogenDataset(dataset, 'atoms', 'coordinates', True, True)
        dataset = RemoveHydrogenDataset(dataset, 'pocket_atoms', 'pocket_coordinates', True, True)
        dataset = NormalizeDockingPoseDataset(dataset, 'coordinates', 'pocket_coordinates', center_coordinates='center_coordinates')
        mol_dataset = CroppingDataset(dataset, self.seed, 'atoms', 'coordinates', self.args.max_atoms)
        pocket_dataset = CroppingDataset(dataset, self.seed, 'pocket_atoms', 'pocket_coordinates', self.args.max_pocket_atoms)


        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = KeyDataset(mol_dataset, 'atoms')
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = KeyDataset(mol_dataset, 'coordinates')
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(pocket_dataset, 'pocket_atoms')
        src_pocket_dataset = TokenizeDataset(src_pocket_dataset, self.pocket_dictionary, max_seq_len=self.args.max_seq_len)
        src_pocket_dataset = PrependAndAppend(src_pocket_dataset, self.pocket_dictionary.bos(), self.pocket_dictionary.eos())
        pocket_edge_type = EdgeTypeDataset(src_pocket_dataset, len(self.pocket_dictionary))
        coord_pocket_dataset = KeyDataset(pocket_dataset, 'pocket_coordinates')
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(distance_pocket_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "mol_src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "mol_src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                        "pocket_src_tokens": RightPadDataset(
                            src_pocket_dataset,
                            pad_idx=self.pocket_dictionary.pad(),
                        ),
                        "pocket_src_coord": RightPadDatasetCoord(
                            coord_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_distance": RightPadDataset2D(
                            distance_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_edge_type": RightPadDataset2D(
                            pocket_edge_type,
                            pad_idx=0,
                        ),
                    },
                    "target": {
                        "all_target": RawLabelDataset(tgt_dataset), 
                    },
                    "system": RawArrayDataset(
                        system_dataset
                    ),
                    "confid": RawArrayDataset(
                        confid_dataset
                    ),
                },
            )
        if split.startswith('train'):
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)
        if args.only_fc:
            freeze_by_names(model, layer_names=['mol_model','pocket_model'])
        return model

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)
