python ./unimol/infer.py --user-dir ./unimol ./examples/ifd_scoring --valid-subset test \
       --task-name 'competition' \
       --results-path ./examples/ifd_scoring \
       --num-workers 8 --ddp-backend=c10d \
       --task ifd_scoring --loss ifd_scoring --arch ifd_scoring --batch-size 32 \
       --path ./IFD_run_lr_3e-4_bs_32_epoch_10_wp_0.06_fold_0/checkpoint_best.pt \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple