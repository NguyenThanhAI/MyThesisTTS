import os
import argparse

from preprocessor_lmdb import LMDBPreprocessor, LMDBDurPitchEnergyEmbedProcessor

def str2bool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="LJSpeech", help="Name of the dataset to preprocess")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\TTS_Raw_Dataset", help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default=r"D:\TTS_Preprocessed_Grad_TTS\Phoneme_Mel_Pitch_Energy_Speaker_Embed", help="Directory to save the LMDB files")
    parser.add_argument("--estimate_size", type=str2bool, default=True, help="Estimate LMDB size without creating files")
    parser.add_argument("--train_map_size", type=float, default=2.2e9, help="LMDB map size for training set")
    parser.add_argument("--val_map_size", type=float, default=0.12e9, help="LMDB map size for validation set")
    parser.add_argument("--train_val_ratio", type=float, default=0.95, help="Ratio of training to validation data")
    parser.add_argument("--use_dur_pitch_energy_embed", type=str2bool, default=False, help="Whether to use duration, pitch, and energy embedding")
    parser.add_argument("--pitch_feature", type=str, default="phoneme_level", help="Pitch feature level: frame_level or phoneme_level")
    parser.add_argument("--energy_feature", type=str, default="phoneme_level", help="Energy feature level: frame_level or phoneme_level")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    print(f"Arguments")
    
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    estimate_size = args.estimate_size
    train_map_size = args.train_map_size
    val_map_size = args.val_map_size
    train_val_ratio = args.train_val_ratio
    use_dur_pitch_energy_embed = args.use_dur_pitch_energy_embed
    pitch_feature = args.pitch_feature
    energy_feature = args.energy_feature

    if use_dur_pitch_energy_embed:
        preprocessor = LMDBDurPitchEnergyEmbedProcessor(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            save_dir=save_dir,
            estimate_size=estimate_size,
            train_val_ratio=train_val_ratio,
            train_map_size=train_map_size,
            val_map_size=val_map_size,
            pitch_feature=pitch_feature,
            energy_feature=energy_feature,
        )
    else:
        preprocessor = LMDBPreprocessor(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            save_dir=save_dir,
            estimate_size=estimate_size,
            train_val_ratio=train_val_ratio,
            train_map_size=train_map_size,
            val_map_size=val_map_size,
        )

    preprocessor.build_from_path()
    preprocessor.get_total_size()