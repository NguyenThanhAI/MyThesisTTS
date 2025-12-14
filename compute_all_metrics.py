import os
import argparse
import json
import numpy as np
from metrics.metrics_calculator import MetricCalculator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesized_data_dir", type=str, required=False, default=r"D:\Synthesized_Wavs_Directory", help="Path to the synthesized data directory.")
    parser.add_argument("--model_name", type=str, required=False, default="grad_tts_multi_speaker_LJSpeech_use_saln_steps_1167000", help="Name of the model.")
    parser.add_argument("--dataset_name", type=str, required=False, default="LJSpeech", help="Name of the dataset.")
    parser.add_argument("--reference_data_dir", type=str, required=False, default=r"D:\TTS_Raw_Dataset", help="Path to the reference data directory.")
    args = parser.parse_args()

    synthesized_data_dir = args.synthesized_data_dir
    model_name = args.model_name
    dataset_name = args.dataset_name
    reference_data_dir = args.reference_data_dir

    metrics_cal = MetricCalculator(
        synthetized_data_dir=synthesized_data_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        reference_data_dir=reference_data_dir
    )

    metrics_cal.get_all_metrics()
