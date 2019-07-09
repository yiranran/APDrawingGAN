set -ex
python train.py --dataroot dataset/data --name formal_noinit --use_local --discriminator_local --niter 300 --niter_decay 0 --save_epoch_freq 25