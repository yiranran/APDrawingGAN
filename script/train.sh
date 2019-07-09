set -ex
python train.py --dataroot dataset/data --name formal --continue_train --use_local --discriminator_local --niter 300 --niter_decay 0 --save_epoch_freq 25