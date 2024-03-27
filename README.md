# StackGAN
Unofficial custmized StackGAN.

## python StackGAN.py --help
```
usage: StackGAN.py [-h] [--csv_path CSV_PATH] [--image_dir IMAGE_DIR] [--result_dir RESULT_DIR]
                   [--weight_dir WEIGHT_DIR] [--lr LR] [--mul_lr_dis MUL_LR_DIS] [--batch_size BATCH_SIZE]
                   [--num_train NUM_TRAIN] [--cpu] [--noresume] [--generate GENERATE]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH
  --image_dir IMAGE_DIR
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --lr LR
  --mul_lr_dis MUL_LR_DIS
  --batch_size BATCH_SIZE
  --num_train NUM_TRAIN
  --cpu
  --noresume
  --generate GENERATE
```

## python CLIP.py --help
```
usage: CLIP.py [-h] [--csv_path CSV_PATH] [--image_dir IMAGE_DIR] [--result_dir RESULT_DIR] [--weight_dir WEIGHT_DIR]
               [--CLIP_max_patches CLIP_MAX_PATCHES] [--CLIP_patch_size CLIP_PATCH_SIZE]
               [--CLIP_sentence_size CLIP_SENTENCE_SIZE] [--lr LR] [--batch_size BATCH_SIZE] [--num_train NUM_TRAIN]
               [--cpu] [--noresume]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH
  --image_dir IMAGE_DIR
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --CLIP_max_patches CLIP_MAX_PATCHES
  --CLIP_patch_size CLIP_PATCH_SIZE
  --CLIP_sentence_size CLIP_SENTENCE_SIZE
  --lr LR
  --batch_size BATCH_SIZE
  --num_train NUM_TRAIN
  --cpu
  --noresume
```

## python CLIP-StackGAN.py --help
```
usage: CLIP-StackGAN.py [-h] [--csv_path CSV_PATH] [--image_dir IMAGE_DIR] [--result_dir RESULT_DIR]
                        [--weight_dir WEIGHT_DIR] [--CLIP_max_patches CLIP_MAX_PATCHES]
                        [--CLIP_patch_size CLIP_PATCH_SIZE] [--CLIP_sentence_size CLIP_SENTENCE_SIZE]
                        [--aug_threshold AUG_THRESHOLD] [--lr LR] [--mul_lr_dis MUL_LR_DIS] [--batch_size BATCH_SIZE]
                        [--num_train NUM_TRAIN] [--cpu] [--noresume] [--generate GENERATE]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH
  --image_dir IMAGE_DIR
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --CLIP_max_patches CLIP_MAX_PATCHES
  --CLIP_patch_size CLIP_PATCH_SIZE
  --CLIP_sentence_size CLIP_SENTENCE_SIZE
  --aug_threshold AUG_THRESHOLD
  --lr LR
  --mul_lr_dis MUL_LR_DIS
  --batch_size BATCH_SIZE
  --num_train NUM_TRAIN
  --cpu
  --noresume
  --generate GENERATE
```

**Note:**
- resume.pkl is a file that saves learning checkpoints for resume and includes models, weight data, etc.
- If a weight.pth file exists in the current directory, the network weights will be automatically read.
