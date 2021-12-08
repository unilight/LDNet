# LDNet: Baeline system of the first VoiceMOS challenge

Author: Wen-Chin Huang (Nagoya University)
Email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

The LDNet system implemented in this repository serves as one of the baselines of the first VoiceMOS challenge, a challenge to compare different systems and approaches on the task of predicting the MOS score of synthetic speech. In this challenge, we use the BVCC dataset. 

## Training phase (phase 1)

During the training phase, the training set and the developement set are released. In the following, we demonstrate how to train the model using the training set, and decode using the developement set to generate a result file that can be submitted to the CodaLab platform.

### Data preparation

After downloading the dataset preparation scripts, please follow the instructions to gather the complete training and development set. For the rest of this README, we assume that the data is put under `data/`, but feel free to put it somewhere else. The data directorty should have the following structure:
```
data
└── phase1-main
    ├── DATA
    │   ├── mydata_system.csv
    │   ├── sets
    │   │   ├── DEVSET
    │   │   ├── train_mos_list.txt
    │   │   ├── TRAINSET
    │   │   └── val_mos_list.txt
    │   └── wav
    └─── ...
```

### Training

According to the [results](./imgs/results.png), `LDNet-ML` gives the best results. We use the `LDNet-ML_MobileNetV3_FFN_1e-3.yaml` config with seed `2337`:

```
python train.py --dataset_name BVCC --data_dir ./data/phase1-main/DATA --config configs/LDNet-ML_MobileNetV3_FFN_1e-3.yaml --update_freq 2 --seed 2337 --tag <tag_name>
```

### Inference for Main Track

After the training ends, we can do inference using the saved checkpoints one-by-one:

```
python inference_for_voicemos.py --tag LDNet-ML_MobileNetV3_FFN_1e-3 --mode mean_listener
```

All results will be saved in `exp/<tag>/BVCC_mean_listener_valid`, including some figures for inspection (see [here](./README#Inference) for more details) and a file named `<epoch>_answer.txt`. A summary of the results will also be saved in `exp/<tag>/BVCC_mean_listener.csv`. We can use this file to choose the checkpoint we want to use.

### Inference for OOD Track

TODO

### Submission to CodaLab

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called `answer.txt` (this naming is a **MUST**).  
To submit to the CodaLab competition platform, choose one specific checkpoint answer file (i.e. `<epoch>_answer.txt`) and rename it to `answer.txt`. Then, compress it with zip format (via `zip` command in Linux or GUI in MacOS) and name it whatever you want. Then this zip file is ready to be submitted!

### Pretrained model

We provide a pretrained model under `exp/Pretrained-LDNet-ML-2337`. To perform inference with it:

```
python inference_for_voicemos.py --tag Pretrained-LDNet-ML-2337 --mode mean_listener
```

And here is what you should get:
```
[Info] Number of valid samples: 1066
[Info] Model parameters: 957057
=================================================
[Info] Evaluating ep 27000
100%|██████████████| 1066/1066 [00:16<00:00, 66.50it/s]
[UTTERANCE] valid MSE: 0.318, LCC: 0.785, SRCC: 0.787, KTAU: 0.597
[SYSTEM] valid MSE: 0.110, LCC: 0.925, SRCC: 0.918, KTAU: 0.757
```

## Evaluation phase (phase 2)

Will be updated once we enter phase 2!