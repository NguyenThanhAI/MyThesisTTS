import os

import pickle
import numpy as np

import torch
import torch.utils
import torch.utils.data

from model.utils import fix_len_compatibility

from pitch_utils import norm_interp_f0, get_lf0_cwt


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
        duration_path = os.path.join(self.data_dir, self.dataset_name, "duration", "{}.npy".format(os.path.splitext(file_name)[0]))
        mel_path = os.path.join(self.data_dir, self.dataset_name, "mel", "{}.npy".format(os.path.splitext(file_name)[0]))
        pitch_path = os.path.join(self.data_dir, self.dataset_name, "pitch", "{}.npy".format(os.path.splitext(file_name)[0]))
        energy_path = os.path.join(self.data_dir, self.dataset_name, "energy", "{}.npy".format(os.path.splitext(file_name)[0]))
        duration = np.load(duration_path)
        duration = torch.FloatTensor(duration)
        mel = np.load(mel_path)
        mel = torch.FloatTensor(mel)
        pitch = np.load(pitch_path)
        pitch = torch.FloatTensor(pitch)
        energy = np.load(energy_path)
        energy = torch.FloatTensor(energy)

        assert pitch.shape[0] == energy.shape[0]

        item = {"x": phoneme_sequence, "y": mel, "duration": duration, "pitch": pitch, "energy": energy}

        return item

    def __len__(self):
        return len(self.info_file_content)
    
    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch
    

class PrecomputedTextMelDurPitchBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        # print(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        duration = torch.zeros((B, x_max_length), dtype=torch.float32)
        pitch = torch.zeros((B, 1, x_max_length), dtype=torch.float32)
        energy = torch.zeros((B, 1, x_max_length), dtype=torch.float32)

        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, duration_, pitch_, energy_ = item["y"], item["x"], item["duration"], item["pitch"], item["energy"]
            
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])

            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

            duration[i, :duration_.shape[-1]] = duration_
            pitch[i, 0, :pitch_.shape[-1]] = pitch_
            energy[i, 0, :energy_.shape[-1]] = energy_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)

        return {"x": x, "x_lengths": x_lengths, 
                "y": y, "y_lengths": y_lengths,
                "duration": duration, 
                "pitch": pitch, "energy": energy}


class PreComputedTextMelDurPitchwithCWTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, dataset_name: str, pitch_type: str="cwt", is_train: bool=True):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.get_data_info()
        self.pitch_type = pitch_type

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
        duration_path = os.path.join(self.data_dir, self.dataset_name, "duration", "{}.npy".format(os.path.splitext(file_name)[0]))
        mel_path = os.path.join(self.data_dir, self.dataset_name, "mel", "{}.npy".format(os.path.splitext(file_name)[0]))
        pitch_path = os.path.join(self.data_dir, self.dataset_name, "pitch", "{}.npy".format(os.path.splitext(file_name)[0]))
        energy_path = os.path.join(self.data_dir, self.dataset_name, "energy", "{}.npy".format(os.path.splitext(file_name)[0]))
        mel2ph_path = os.path.join(self.data_dir, self.dataset_name, "mel2ph", "{}.npy".format(os.path.splitext(file_name)[0]))
        f0_path = os.path.join(self.data_dir, self.dataset_name, "f0", "{}.npy".format(os.path.splitext(file_name)[0]))
        cwt_spec_path = os.path.join(self.data_dir, self.dataset_name, "cwt_specs", "{}.npy".format(os.path.splitext(file_name)[0]))
        cwt_scales_path = os.path.join(self.data_dir, self.dataset_name, "cwt_scales", "{}.npy".format(os.path.splitext(file_name)[0]))
        f0cwt_mean_std_path = os.path.join(self.data_dir, self.dataset_name, "f0cwt_mean_std", "{}.npy".format(os.path.splitext(file_name)[0]))
        duration = np.load(duration_path)
        duration = torch.FloatTensor(duration)
        mel = np.load(mel_path)
        mel = torch.FloatTensor(mel)
        pitch = np.load(pitch_path)
        pitch = torch.FloatTensor(pitch)
        energy = np.load(energy_path)
        energy = torch.FloatTensor(energy)
        mel2ph = np.load(mel2ph_path)
        mel2ph = torch.IntTensor(mel2ph)
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0=f0,
                                use_uv=True)
        
        cwt_spec = f0_mean = f0_std = f0_ph = None
        if self.pitch_type == "cwt":
            cwt_spec_path = os.path.join(self.data_dir, self.dataset_name, "cwt_specs", "{}.npy".format(os.path.splitext(file_name)[0]))
            cwt_spec = np.load(cwt_spec_path)
            f0cwt_mean_std_path = os.path.join(self.data_dir, self.dataset_name, "f0cwt_mean_std", "{}.npy".format(os.path.splitext(file_name)[0]))
            f0cwt_mean_std = np.load(f0cwt_mean_std_path)
            f0_mean = float(f0cwt_mean_std[0])
            f0_std = float(f0cwt_mean_std[1])
        elif self.pitch_type == "ph":
            f0_phlevel_sum = torch.zeros(phoneme_sequence.shape).float().scatter_add(0, mel2ph.long() - 1, f0.float())
            f0_phlevel_num = torch.zeros(phoneme_sequence.shape).float().scatter_add(0, mel2ph.long() - 1, torch.ones_like(f0.float()))
            f0_ph = f0_phlevel_sum / (f0_phlevel_num + 1e-6)

        # assert pitch.shape[0] == energy.shape[0]

        item = {"x": phoneme_sequence, 
                "y": mel, 
                "duration": duration, 
                "pitch": pitch, 
                "energy": energy,
                "f0": f0,
                "f0_ph": f0_ph,
                "uv": uv,
                "cwt_spec": cwt_spec,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "mel2ph": mel2ph}

        return item

    def __len__(self):
        return len(self.info_file_content)
    

class PreComputedTextMelDurPitchwithCWTDatasetBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        y_max_length = max([item["y"].shape[-1] for item in batch])
        duration_max_length = max([item["duration"].shape[-1] for item in batch])
        pitch_max_length = max([item["pitch"].shape[-1] for item in batch])
        energy_max_length = max([item["energy"].shape[-1] for item in batch])
        f0_max_length = max([item["f0"].shape[-1] for item in batch])
        f0_ph_max_length = max([item["f0_ph"].shape[-1] if item["f0_ph"] is not None else 0 for item in batch])
        uv_max_length = max([item["uv"].shape[-1] for item in batch])
        cwt_spec_max_length = max([item["cwt_spec"].shape[0] if item["cwt_spec"] is not None else 0 for item in batch])
        mel2ph_max_length = max([item["mel2ph"].shape[-1] for item in batch])

        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        duration = torch.zeros((B, duration_max_length), dtype=torch.float32)
        pitch = torch.zeros((B, 1, pitch_max_length), dtype=torch.float32)
        energy = torch.zeros((B, 1, energy_max_length), dtype=torch.float32)
        f0 = torch.zeros((B, 1, f0_max_length), dtype=torch.float32)
        if f0_ph_max_length == 0:
            f0_ph_max_length = None
            f0_ph = None
        else:
            f0_ph = torch.zeros((B, 1, f0_ph_max_length), dtype=torch.float32)
        uv = torch.zeros((B, 1, uv_max_length), dtype=torch.float32)
        if cwt_spec_max_length == 0:
            cwt_spec = None
        else:
            cwt_spec = torch.zeros((B, cwt_spec_max_length, batch[0]["cwt_spec"].shape[-1]), dtype=torch.float32)
        mel2ph = np.zeros((B, mel2ph_max_length), dtype=np.int64)

        return {"x": x, "x_lengths": torch.LongTensor([item["x"].shape[-1] for item in batch]),
                "y": y, "y_lengths": torch.LongTensor([item["y"].shape[-1] for item in batch]),
                "duration": duration,  "duration_lengths": torch.LongTensor([item["duration"].shape[-1] for item in batch]),
                "pitch": pitch, "pitch_lengths": torch.LongTensor([item["pitch"].shape[-1] for item in batch]),
                "energy": energy, "energy_lengths": torch.LongTensor([item["energy"].shape[-1] for item in batch]),
                "f0": f0, "f0_lengths": torch.LongTensor([item["f0"].shape[-1] for item in batch]),
                "f0_ph": f0_ph, "f0_ph_lengths": torch.LongTensor([item["f0_ph"].shape[-1] if item["f0_ph"] is not None else 0 for item in batch]),
                "uv": uv, "uv_lengths": torch.LongTensor([item["uv"].shape[-1] for item in batch]),
                "cwt_spec": cwt_spec, "cwt_spec_lengths": torch.LongTensor([item["cwt_spec"].shape[0] if item["cwt_spec"] is not None else 0 for item in batch]), "cwt_spec_lengths": cwt_spec_max_length,
                "f0_mean": torch.FloatTensor([item["f0_mean"] for item in batch]),
                "f0_std": torch.FloatTensor([item["f0_std"] for item in batch]),
                "mel2ph": torch.LongTensor(mel2ph), "mel2ph_lengths": torch.LongTensor([item["mel2ph"].shape[-1] for item in batch])}
        

if __name__ == "__main__":
    data_dir = r"D:\TTS_Preprocessed"
    dataset_name = "LJSpeech"
    batch_size = 2

    # dataset = PrecomputedTextMelDurPitchDataset(data_dir=data_dir,
    #                                             dataset_name=dataset_name,
    #                                             is_train=True)
    
    # collate = PrecomputedTextMelDurPitchBatchCollate()

    # dataloader = torch.utils.data.DataLoader(dataset=dataset,
    #                                          batch_size=batch_size,
    #                                          collate_fn=collate)
    
    # for batch in dataloader:
    #     print(batch)

    dataset = PreComputedTextMelDurPitchwithCWTDataset(data_dir=data_dir,
                                                       dataset_name=dataset_name,
                                                       pitch_type="cwt")
    
    for el in dataset:
        print(f"Phoneme Sequence: {el['x'].shape}, mel shape: {el['y'].shape}, "
              f"duration shape: {el['duration'].shape}, "
              f"pitch shape: {el['pitch'].shape}, "
              f"energy shape: {el['energy'].shape}, "
              f"f0 shape: {el['f0'].shape}, "
              f"f0_ph shape: {el['f0_ph'].shape if el['f0_ph'] is not None else None}, "
              f"uv shape: {el['uv'].shape}, "
              f"cwt_spec shape: {el['cwt_spec'].shape if el['cwt_spec'] is not None else None}, "
              f"f0_mean: {el['f0_mean']}, "
              f"f0_std: {el['f0_std']}, "
              f"mel2ph shape: {el['mel2ph'].shape}")
        break

    collate = PreComputedTextMelDurPitchwithCWTDatasetBatchCollate()

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             collate_fn=collate)
    
    for batch in dataloader:
        print(batch)