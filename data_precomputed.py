import os

import pickle
import numpy as np

import torch
import torch.utils
import torch.utils.data

from model.utils import fix_len_compatibility


class PrecomputedTextMelDurPitchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, dataset_name: str, is_train: bool=True):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.get_data_info()

    def get_data_info(self):
        info_file = os.path.join(self.data_dir, self.dataset_name, "dataset_info.pkl")
        with open(info_file, "rb") as f:
            if self.is_train:
                self.info_file_content = pickle.load(f)["train"]
            else:
                self.info_file_content = pickle.load(f)["valid"]

    def __getitem__(self, index):
        element_info = self.info_file_content[index]
        phoneme_sequence = element_info["script"]
        phoneme_sequence = torch.IntTensor(phoneme_sequence)
        file_name = element_info["file"]
        mel_path = os.path.join(self.data_dir, self.dataset_name, "mel", "{}.npy".format(os.path.splitext(file_name)[0]))
        pitch_path = os.path.join(self.data_dir, self.dataset_name, "pitch", "{}.npy".format(os.path.splitext(file_name)[0]))
        energy_path = os.path.join(self.data_dir, self.dataset_name, "energy", "{}.npy".format(os.path.splitext(file_name)[0]))
        mel = np.load(mel_path)
        mel = torch.FloatTensor(mel)
        pitch = np.load(pitch_path)
        pitch = torch.FloatTensor(pitch)
        energy = np.load(energy_path)
        energy = torch.FloatTensor(energy)

        assert mel.shape[1] == pitch.shape[0] and pitch.shape[0] == energy.shape[0]

        item = {"x": phoneme_sequence, "y": mel, "pitch": pitch, "energy": energy}

        return item

    def __len__(self):
        return len(self.info_file_content)
    

class PrecomputedTextMelDurPitchBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        print(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        pitch = torch.zeros((B, 1, y_max_length), dtype=torch.float32)
        energy = torch.zeros((B, 1, y_max_length), dtype=torch.float32)

        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, pitch_, energy_ = item["y"], item["x"], item["pitch"], item["energy"]
            
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])

            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

            pitch[i, 0, :pitch_.shape[-1]] = pitch_
            energy[i, 0, :energy_.shape[-1]] = energy_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)

        return {"x": x, "x_lengths": x_lengths, 
                "y": y, "y_lengths": y_lengths, 
                "pitch": pitch, "energy": energy}


if __name__ == "__main__":
    data_dir = r"D:\TTS_Preprocessed"
    dataset_name = "LJSpeech"
    batch_size = 2

    dataset = PrecomputedTextMelDurPitchDataset(data_dir=data_dir,
                                                dataset_name=dataset_name,
                                                is_train=True)
    
    collate = PrecomputedTextMelDurPitchBatchCollate()

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             collate_fn=collate)
    
    for batch in dataloader:
        print(batch)