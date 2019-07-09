## Training/test Tips
- Flags: see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags. The default values of these options are somtimes adjusted in the model files.

- CPU/GPU (default `--gpu_ids 0`): set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batch_size 32`) to benefit from multiple GPUs.

- Visualization: during training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will appear on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have `visdom` installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id -1`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.

- Fine-tuning/Resume training: to fine-tune a pre-trained model, or resume the previous training, use the `--continue_train` flag. The program will then load the model based on `which_epoch`. By default, the program will initialize the epoch count as 1. Set `--epoch_count <int>` to specify a different starting epoch count.
