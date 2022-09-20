### gpu configs
n_gpu=1
MASTER_PORT=10086

###
data_path='./examples/ifd_scoring'
finetune_mol_model='./weights/mol_pre_no_h_220816.pt'
finetune_pocket_model='./weights/pocket_pre_220816.pt'
exp_name='IFD_run_v2'

###
lr=3e-4
batch_size=8
epoch=10
dropout=0.2
warmup=0.06
update_freq=4
valid_fold=0
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
save_dir="${exp_name}_lr_${lr}_bs_${global_batch_size}_epoch_${epoch}_wp_${warmup}_fold_${valid_fold}"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task-name 'competition' \
       --task ifd_scoring --loss ifd_scoring --arch ifd_scoring  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --fold $valid_fold \
       --mol-pooler-dropout $dropout --pocket-pooler-dropout $dropout \
       --update-freq $update_freq --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --patience 2000 --all-gather-list-size 102400 \
       --finetune-mol-model $finetune_mol_model \
       --finetune-pocket-model $finetune_pocket_model \
       --save-dir $save_dir \
       --find-unused-parameters \
       --required-batch-size-multiple 1 \
       --best-checkpoint-metric valid_loss \
       --keep-best-checkpoints 1