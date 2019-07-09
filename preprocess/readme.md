## Preprocessing steps

Face photos (and paired drawings) need to be aligned and have background mask detected. Aligned images, facial lamdmark files (txt) and background masks are needed for training and testing.

### 1. Align, resize, crop images to 512x512 and prepare facial landmarks

All training and testing images in our model are aligned using facial landmarks. And landmarks after alignment are needed in our code.

- First, 5 facial landmark for a face photo need to be detected (we detect using [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)(MTCNNv1)).

- Then, we provide a matlab function in `face_align_512.m` to align, resize and crop face photos (and corresponding drawings) to 512x512.Call this function in MATLAB to align the image to 512x512.
For example, for `img_1701.jpg` in `example` dir, 5 detected facial landmark is saved in `example/img_1701_facial5point.mat`. Call following in MATLAB:
```bash
load('example/img_1701_facial5point.mat');
[trans_img,trans_facial5point]=face_align_512('example/img_1701.jpg',facial5point,'example');
```

This will align the image and output aligned image and transformed facial landmark (in txt format) in `example` folder.
See `face_align_512.m` for more instructions.

- The saved transformed facial landmark need to be copied to `lm_dir` (see [base flags](../options/base_options.py), default is `dataset/landmark/ALL`), and has the **same filename** with aligned face photos (e.g. `dataset/data/test_single/31.png` should have landmark file `dataset/landmark/ALL/31.txt`).

### 2. Prepare background masks

Background masks are needed in our code.

In our work, background mask is segmented by method in
"Automatic Portrait Segmentation for Image Stylization"
Xiaoyong Shen, Aaron Hertzmann, Jiaya Jia, Sylvain Paris, Brian Price, Eli Shechtman, Ian Sachs. Computer Graphics Forum, 35(2)(Proc. Eurographics), 2016.

- We use code in http://xiaoyongshen.me/webpage_portrait/index.html to detect background masks for face photos.
A sample background mask is shown in `example/img_1701_aligned_bgmask.png`.

- The background masks need to be copied to `bg_dir` (see [base flags](../options/base_options.py), default is `dataset/mask/ALL`), and has the **same filename** with aligned face photos (e.g. `dataset/data/test_single/31.png` should have background mask `dataset/mask/ALL/31.png`)  


### 3. (For training) Prepare more training data

We provide a python script to generate training data in the form of pairs of images {A,B}, i.e. pairs {face photo, drawing}. This script will concatenate each pair of images horizontally into one single image. Then we can learn to translate A to B:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `test`, etc. In `/path/to/data/A/train`, put training face photos. In `/path/to/data/B/train`, put the corresponding artist drawings. Repeat same for `test`.

Corresponding images in a pair {A,B} must both be images after aligning and of size 512x512, and have the same filename, e.g., `/path/to/data/A/train/1.png` is considered to correspond to `/path/to/data/B/train/1.png`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.