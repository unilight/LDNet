# LDNet: Baeline system of the first VoiceMOS challenge

Author: Wen-Chin Huang (Nagoya University)
Email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

The LDNet system implemented in this repository serves as one of the baselines of the first VoiceMOS challenge, a challenge to compare different systems and approaches on the task of predicting the MOS score of synthetic speech.

## Training phase (phase 1)

During the training phase, participants should receive datasets for the main (BVCC) track and the OOD track. Each dataset will contain the training set and the developement set. In this instruction, we demonstrate how to perform:

- Training a LDNet from scratch with the BVCC training set.
- Inference on the BVCC validation set with the trained LDNet.
- Zero-shot inference on the OOD validation set with the LDNet trained on the BVCC training set.
- Fine-tuning on the OOD labeled training set with with the LDNet pretrained on the BVCC training set.
- Inference on the OOD validation set with the LDNet fine-tuned on the OOD labeled training set.
- Submission generation for the CodaLab platform.

### Data preparation for both tracks

After downloading the dataset preparation scripts for both tracks, please follow the instructions to gather the complete datasets. For the rest of this README, we assume that the data is put under `data/`, but feel free to put it somewhere else. The data directorty should have the following structure:
```
data
├── phase1-main
│    ├── DATA
│    │   ├── mydata_system.csv
│    │   ├── sets
│    │   │   ├── DEVSET
│    │   │   ├── train_mos_list.txt
│    │   │   ├── TRAINSET
│    │   │   └── val_mos_list.txt
│    │   └── wav
│    └─── ...
└── phase1-ood
     ├── DATA
     │   ├── mydata_system.csv
     │   ├── sets
     │   │   ├── DEVSET
     │   │   ├── train_mos_list.txt
     │   │   ├── TRAINSET
     │   │   ├── unlabeled_mos_list.txt
     │   │   └── val_mos_list.txt     
     │   └── wav
     └── ...
```

### Pretrained model

We provide two pretrained models. Feel free to use these pretrained models to test the inference code and generate sample submission files to familizarize yourself with CodaLab.

- `exp/Pretrained-LDNet-ML-2337`: model trained on the BVCC training set with the `LDNet-ML_MobileNetV3_FFN_1e-3.yaml` config and seed `2337`.
- `exp/FT_LDNet-ML-bs20`: model fine-tuned on the OOD labeled training set with the `LDNet-ML_MobileNetV3_FFN_1e-3_ft.yaml` config and seed `1337`.

### Training a LDNet from scratch with the BVCC training set.

Although you can use any config mentioned in the [main README](./README#Training), according to the [results](./imgs/results.png), `LDNet-ML` gives the best results. So, here we demonstrate how to use the `LDNet-ML_MobileNetV3_FFN_1e-3.yaml` config with seed `2337` to perform training:

```
python train.py --dataset_name BVCC --data_dir ./data/phase1-main/DATA --config configs/LDNet-ML_MobileNetV3_FFN_1e-3.yaml --update_freq 2 --seed 2337 --tag <tag_name>
```

All checkpoints will be saved in, by default, `exp/<tag_name>/model-<step>.pt`

### Inference on the BVCC validation set with the trained LDNet.

After the training ends, we can do inference using the saved checkpoints one-by-one:

```
python inference_for_voicemos.py --dataset_name BVCC --data_dir data/phase1-main/DATA --tag <tag_name> --mode mean_listener
```

This is what can be expected if using the `Pretrained-LDNet-ML-2337` model:

```
[Info] Number of valid samples: 1066
[Info] Model parameters: 957057
=================================================
[Info] Evaluating ep 27000
100%|██████████████| 1066/1066 [00:16<00:00, 66.50it/s]
[UTTERANCE] valid MSE: 0.318, LCC: 0.785, SRCC: 0.787, KTAU: 0.597
[SYSTEM] valid MSE: 0.110, LCC: 0.925, SRCC: 0.918, KTAU: 0.757
```

All results will be saved in `exp/<tag_name>/BVCC_mean_listener_valid/`, including some figures for inspection (see [here](./README#Inference) for more details) and a file named `<step>_answer.txt`. A summary of the results will also be saved in `exp/<tag_name>/BVCC_mean_listener.csv`. We can use this file to manually choose which checkpoint we want to use.

### Zero-shot inference on the OOD validation set with the LDNet trained on the BVCC training set.

We can simply change the dataset to perform zero-shot inference on the OOD validation set:

```
python inference_for_voicemos.py --dataset_name OOD --data_dir data/phase1-ood/DATA --tag <tag_name> --mode mean_listener
```

Then, all results will be saved in `exp/<tag_name>/OOD_mean_listener_valid/`.

### Fine-tuning on the OOD labeled training set with with the LDNet pretrained on the BVCC training set

We can perform a very simple fine-tuning procedure on the OOD labeled training set with a model pretrained on the BVCC training set. Here we demonstrate how to use the `LDNet-ML_MobileNetV3_FFN_1e-3_ft.yaml` config with the `Pretrained-LDNet-ML-2337` pretrained model to perform fine-tuning:

```
python train.py --dataset_name OOD --data_dir data/phase1-ood/DATA --config configs/LDNet-ML_MobileNetV3_FFN_1e-3_ft.yaml --tag <tag_name> --pretrained_model_path exp/Pretrained-LDNet-ML-2337/model-27000.pt
```

All checkpoints will be saved in, again, by default, `exp/<tag_name>/model-<step>.pt`

### Inference on the OOD validation set with the LDNet fine-tuned on the OOD labeled training set.

Similar to above, we can simply change the dataset to perform inference on the OOD validation set using the fine-tuned model:

```
python inference_for_voicemos.py --dataset_name OOD --data_dir data/phase1-ood/DATA --tag <tag_name> --mode mean_listener
```

Then, all results will be saved in `exp/<tag_name>/OOD_mean_listener_valid/`.

### Submission generation for the CodaLab platform

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called `answer.txt` (this naming is a **MUST**). We have prepared a convenient packing script. If you only want to submit the main track result, pass the zip file name as the first argument, and pass the main track answer file as the the second argument. The following is an example:

```
./pack_for_voicemos.sh <any_name>.zip exp/<tag_name>/BVCC_<inference_mode>_<split>/<step>_answer.txt
```

If you want to submit results for both the main track and the OOD track, pass the OOD track answer as the third argument:

```
./pack_for_voicemos.sh <any_name>.zip exp/<tag_name>/BVCC_<inference_mode>_<split>/<step>_answer.txt exp/<tag_name>/OOD_<inference_mode>_<split>/<step>_answer.txt
```

Then, submit the generated `<any_name>.zip` to the CodaLab platform!

## Evaluation phase (phase 2)

Will be updated once we enter phase 2!