import os
import librosa
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import scipy

from collections import defaultdict
import h5py
from torch.nn.utils.rnn import pad_sequence

FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1

class ASVBC19Dataset(Dataset):
    def __init__(self, original_metadata, data_dir, idtable_path=None, split="train", padding_mode="zero_padding", use_mean_listener=False):
        self.data_dir = data_dir
        self.split = split
        self.padding_mode = padding_mode
        self.use_mean_listener = use_mean_listener

        # add mean listener to metadata
        if use_mean_listener:
            mean_listener_metadata = self.gen_mean_listener_metadata(original_metadata)
            metadata = original_metadata + mean_listener_metadata
        else:
            metadata = original_metadata

        # get judge id table and number of judges
        if idtable_path is not None:
            if os.path.isfile(idtable_path):
                self.idtable = torch.load(idtable_path)
            elif self.split == "train":
                self.gen_idtable(metadata, idtable_path)
            self.num_judges = len(self.idtable)

        self.metadata = []
        if self.split == "train":
            #(NOTE) unlight (210921): need to fix this in the future if we want to do training.
            for wav_name, judge_name, avg_score, score in metadata:
                self.metadata.append([wav_name, avg_score, score, self.idtable[judge_name]])
        else:
            for item in metadata:
                self.metadata.append(item)

            # build system list
            self.systems = list(set([item[0] for item in metadata]))
            
    def __getitem__(self, idx):
        if self.split == "train":
            wav_name, avg_score, score, judge_id = self.metadata[idx]
        else:
            sys_name, wav_name, avg_score = self.metadata[idx]

        h5_path = os.path.join(self.data_dir, "bin", wav_name.replace(".wav", ".h5"))
        data_file = h5py.File(h5_path, 'r')
        mag_sgram = np.array(data_file['mag_sgram'][:])
        timestep = mag_sgram.shape[0]
        mag_sgram = np.reshape(mag_sgram,(timestep, SGRAM_DIM))

        if self.split == "train":
            return mag_sgram, avg_score, score, judge_id
        else:
            return mag_sgram, avg_score, sys_name
    
    def __len__(self):
        return len(self.metadata)

    def gen_mean_listener_metadata(self, original_metadata):
        assert self.split == "train"
        mean_listener_metadata = []
        wav_names = set()
        for wav_name, _, avg_score, _ in original_metadata:
            if wav_name not in wav_names:
                mean_listener_metadata.append([wav_name, "mean_listener", avg_score, avg_score])
                wav_names.add(wav_name)
        return mean_listener_metadata

    def gen_idtable(self, metadata, idtable_path):
        self.idtable = {}
        count = 0
        for _, judge_name, _, _ in metadata:
            # mean listener always takes the last id
            if judge_name not in self.idtable and not judge_name == "mean_listener":
                self.idtable[judge_name] = count
                count += 1
        if self.use_mean_listener:
            self.idtable["mean_listener"] = count
            count += 1
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])
        bs = len(sorted_batch) # batch_size
        avg_scores = torch.FloatTensor([sorted_batch[i][1] for i in range(bs)])
        mag_sgrams = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        mag_sgrams_lengths = torch.from_numpy(np.array([mag_sgram.size(0) for mag_sgram in mag_sgrams]))
        
        if self.padding_mode == "zero_padding":
            mag_sgrams_padded = pad_sequence(mag_sgrams, batch_first=True)
        elif self.padding_mode == "repetitive":
            max_len = mag_sgrams_lengths[0]
            mag_sgrams_padded = []
            for mag_sgram in mag_sgrams:
                this_len = mag_sgram.shape[0]
                dup_times = max_len // this_len
                remain = max_len - this_len * dup_times
                to_dup = [mag_sgram for t in range(dup_times)]
                to_dup.append(mag_sgram[:remain, :])
                mag_sgrams_padded.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
            mag_sgrams_padded = torch.stack(mag_sgrams_padded, dim = 0)
        else:
            raise NotImplementedError

        if not self.split == "train":
            sys_names = [sorted_batch[i][2] for i in range(bs)]
            return mag_sgrams_padded, avg_scores, sys_names
        else:
            scores = torch.FloatTensor([sorted_batch[i][2] for i in range(bs)])
            judge_ids = torch.LongTensor([sorted_batch[i][3] for i in range(bs)])
            return mag_sgrams_padded, mag_sgrams_lengths, avg_scores, scores, judge_ids


class VCC18Dataset(Dataset):
    def __init__(self, original_wav_file, original_score_csv, idtable_path=None, split="traing", use_mean_listener=False):
        self.split = split
        self.use_mean_listener = use_mean_listener

        self.features = {}
        
        # add mean listener to metadata
        if use_mean_listener:
            mean_listener_wav_file, mean_listener_score_csv = self.gen_mean_listener_metadata(original_wav_file, original_score_csv)
            self.wavs = original_wav_file + mean_listener_wav_file
            self.scores = original_score_csv.append(mean_listener_score_csv, ignore_index = True)
        else:
            self.wavs = original_wav_file
            self.scores = original_score_csv

        # get judge id table and number of judges
        if idtable_path is not None:
            if os.path.isfile(idtable_path):
                self.idtable = torch.load(idtable_path)
            elif self.split == "train":
                self.gen_idtable(idtable_path)
            for i, judge_i in enumerate(self.scores['JUDGE']):
                self.scores['JUDGE'][i] = self.idtable[judge_i]
            self.num_judges = len(self.idtable)
        
        # build system list
        self.systems = list(set([name.split("_")[0] for name in self.scores["WAV_PATH"]]))

    def __getitem__(self, idx):
        if type(self.wavs[idx]) == int:
            wav_name = self.wavs[idx - self.wavs[idx]]
        else:
            wav_name = self.wavs[idx]

        # cache features
        if wav_name not in self.features:
            wav, _ = librosa.load(wav_name, sr = 16000)
            feature = np.abs(librosa.stft(wav, n_fft = 512)).T
            self.features[wav_name] = feature
        
        return self.features[wav_name], self.scores['MEAN'][idx], self.scores['MOS'][idx], self.scores['JUDGE'][idx], self.scores["WAV_PATH"][idx].split("_")[0]
    
    def __len__(self):
        return len(self.wavs)
    
    def gen_mean_listener_metadata(self, original_wav_file, original_score_csv):
        assert self.split == "train"
        ks = ["MEAN", "MOS", "JUDGE", "WAV_PATH"]
        mean_listener_wavs = []
        mean_listener_metadata = {k: [] for k in ks}
        for i, line in enumerate(original_wav_file):
            if not type(line) == int and not line in mean_listener_wavs:
                mean_listener_wavs.append(line)
                mean_listener_metadata["MEAN"].append(original_score_csv["MEAN"][i])
                mean_listener_metadata["MOS"].append(original_score_csv["MEAN"][i])
                mean_listener_metadata["WAV_PATH"].append(original_score_csv["WAV_PATH"][i])
                mean_listener_metadata["JUDGE"].append("mean_listener")

        return mean_listener_wavs, pd.DataFrame(mean_listener_metadata)

    def gen_idtable(self, idtable_path):
        self.idtable = {}
        count = 0
        for i, judge_i in enumerate(self.scores['JUDGE']):
            if judge_i not in self.idtable.keys() and not judge_i == "mean_listener":
                self.idtable[judge_i] = count
                count += 1
        if self.use_mean_listener:
            self.idtable["mean_listener"] = count
            count += 1
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, samples):
        # wavs may be list of wave or spectrogram, which has shape (time, feature) or (time,)
        wavs, means, scores, judge_ids, sys_names = zip(*samples)
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        wav_lengths = torch.from_numpy(np.array([wav.shape[0] for wav in wavs]))
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        means = torch.FloatTensor(means)
        scores = torch.FloatTensor(scores)
        
        if not self.split == "train":
            return output_wavs, means, sys_names
        else:
            judge_ids = torch.LongTensor(judge_ids)
            return output_wavs, wav_lengths, means, scores, judge_ids

class BCVCCDataset(Dataset):
    def __init__(self, original_metadata, data_dir, idtable_path=None, split="train", padding_mode="zero_padding", use_mean_listener=False):
        self.data_dir = data_dir
        self.split = split
        self.padding_mode = padding_mode
        self.use_mean_listener = use_mean_listener

        # cache features
        self.features = {}

        # add mean listener to metadata
        if use_mean_listener:
            mean_listener_metadata = self.gen_mean_listener_metadata(original_metadata)
            metadata = original_metadata + mean_listener_metadata
        else:
            metadata = original_metadata

        # get judge id table and number of judges
        if idtable_path is not None:
            if os.path.isfile(idtable_path):
                self.idtable = torch.load(idtable_path)
            elif self.split == "train":
                self.gen_idtable(metadata, idtable_path)
            self.num_judges = len(self.idtable)

        self.metadata = []
        if self.split == "train":
            for wav_name, judge_name, avg_score, score in metadata:
                self.metadata.append([wav_name, avg_score, score, self.idtable[judge_name]])
        else:
            for item in metadata:
                self.metadata.append(item)

            # build system list
            self.systems = list(set([item[0] for item in metadata]))
            
    def __getitem__(self, idx):
        if self.split == "train":
            wav_name, avg_score, score, judge_id = self.metadata[idx]
        else:
            sys_name, wav_name, avg_score = self.metadata[idx]

        # cache features
        if wav_name in self.features:
            mag_sgram = self.features[wav_name]
        else:
            h5_path = os.path.join(self.data_dir, "bin", wav_name + ".h5")
            if os.path.isfile(h5_path):
                data_file = h5py.File(h5_path, 'r')
                mag_sgram = np.array(data_file['mag_sgram'][:])
                timestep = mag_sgram.shape[0]
                mag_sgram = np.reshape(mag_sgram,(timestep, SGRAM_DIM))
            else:
                wav, _ = librosa.load(os.path.join(self.data_dir, "wav", wav_name), sr = 16000)
                mag_sgram = np.abs(librosa.stft(wav, n_fft = 512, hop_length=256, win_length=512, window=scipy.signal.hamming)).astype(np.float32).T
            self.features[wav_name] = mag_sgram

        if self.split == "train":
            return mag_sgram, avg_score, score, judge_id
        else:
            return mag_sgram, avg_score, sys_name, wav_name
    
    def __len__(self):
        return len(self.metadata)

    def gen_mean_listener_metadata(self, original_metadata):
        assert self.split == "train"
        mean_listener_metadata = []
        wav_names = set()
        for wav_name, _, avg_score, _ in original_metadata:
            if wav_name not in wav_names:
                mean_listener_metadata.append([wav_name, "mean_listener", avg_score, avg_score])
                wav_names.add(wav_name)
        return mean_listener_metadata

    def gen_idtable(self, metadata, idtable_path):
        self.idtable = {}
        count = 0
        for _, judge_name, _, _ in metadata:
            # mean listener always takes the last id
            if judge_name not in self.idtable and not judge_name == "mean_listener":
                self.idtable[judge_name] = count
                count += 1
        if self.use_mean_listener:
            self.idtable["mean_listener"] = count
            count += 1
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])
        bs = len(sorted_batch) # batch_size
        avg_scores = torch.FloatTensor([sorted_batch[i][1] for i in range(bs)])
        mag_sgrams = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        mag_sgrams_lengths = torch.from_numpy(np.array([mag_sgram.size(0) for mag_sgram in mag_sgrams]))
        
        if self.padding_mode == "zero_padding":
            mag_sgrams_padded = pad_sequence(mag_sgrams, batch_first=True)
        elif self.padding_mode == "repetitive":
            max_len = mag_sgrams_lengths[0]
            mag_sgrams_padded = []
            for mag_sgram in mag_sgrams:
                this_len = mag_sgram.shape[0]
                dup_times = max_len // this_len
                remain = max_len - this_len * dup_times
                to_dup = [mag_sgram for t in range(dup_times)]
                to_dup.append(mag_sgram[:remain, :])
                mag_sgrams_padded.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
            mag_sgrams_padded = torch.stack(mag_sgrams_padded, dim = 0)
        else:
            raise NotImplementedError

        if not self.split == "train":
            sys_names = [sorted_batch[i][2] for i in range(bs)]
            wav_names = [sorted_batch[i][3] for i in range(bs)]
            return mag_sgrams_padded, avg_scores, sys_names, wav_names
        else:
            scores = torch.FloatTensor([sorted_batch[i][2] for i in range(bs)])
            judge_ids = torch.LongTensor([sorted_batch[i][3] for i in range(bs)])
            return mag_sgrams_padded, mag_sgrams_lengths, avg_scores, scores, judge_ids

def get_dataset(dataset_name, data_dir, split, idtable_path=None, padding_mode="zero_padding", use_mean_listener=False):
    if dataset_name in ["BVCC", "OOD"]:
        names = {"train":"TRAINSET", "valid":"DEVSET", "test":"TESTSET"}
    
        metadata = defaultdict(dict)
        metadata_with_avg = list()

        # read metadata
        with open(os.path.join(data_dir, "sets", names[split]), "r") as f:
            lines = f.read().splitlines()
           
            # line has format <system, wav_name, score, _, judge_name>
            for line in lines:
                parts = line.split(",")
                sys_name = parts[0]
                wav_name = parts[1]
                score = int(parts[2])
                judge_name = parts[4]
                metadata[sys_name + "|" + wav_name][judge_name] = score
        
        # calculate average score
        for _id, v in metadata.items():
            sys_name, wav_name = _id.split("|")
            avg_score = np.mean(np.array(list(v.values())))
            if split == "train":
                for judge_name, score in v.items():
                    metadata_with_avg.append([wav_name, judge_name, avg_score, score])
            else:
                # in testing mode, additionally return system name and only average score
                metadata_with_avg.append([sys_name, wav_name, avg_score])

        return BCVCCDataset(metadata_with_avg, data_dir, idtable_path, split, padding_mode, use_mean_listener)

    elif dataset_name == "vcc2018":
        names = {"train":"vcc2018_training_data.csv", "valid":"vcc2018_valid_data.csv", "test":"vcc2018_testing_data.csv"}
        dataframe = pd.read_csv(os.path.join(data_dir, f'{names[split]}'), index_col=False)
        wavs = []
        filename = ''
        last = 0
        for i in range(len(dataframe)):
            if dataframe['WAV_PATH'][i] != filename:
                wav_name = os.path.join(data_dir, dataframe['WAV_PATH'][i])
                wavs.append(wav_name)
                filename = dataframe['WAV_PATH'][i]
                last = 0
            else:
                last += 1
                wavs.append(last)
        return VCC18Dataset(wavs, dataframe, idtable_path, split, use_mean_listener)

    elif dataset_name in ["asv19", "bc19"]:
        if split == "train":
            raise NotImplementedError

        names = {"train":"train_mos_list.txt", "valid":"val_mos_list.txt", "test":"test_mos_list.txt"}
    
        metadata = defaultdict(dict)
        metadata_with_avg = list()

        # read metadata
        with open(os.path.join(data_dir, "sets", names[split]), "r") as f:
            lines = f.read().splitlines()
           
            # line has format <wav_name, score>
            for line in lines:
                parts = line.split(",")
                wav_name = parts[0]
                sys_name = wav_name.split("-")[1]
                avg_score = float(parts[1])
                if dataset_name == "asv19":
                    avg_score = avg_score * 4.0 / 9.0 + 1.0
                metadata_with_avg.append([sys_name, wav_name, avg_score])

        return ASVBC19Dataset(metadata_with_avg, data_dir, idtable_path, split, padding_mode, use_mean_listener)
    else:
        raise NotImplementedError

def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
