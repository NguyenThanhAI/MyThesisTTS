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
    parser.add_argument("--metrics_list", type=str, nargs='+', default=["wer_un_comma"], help="List of metrics to compute.")
    args = parser.parse_args()

    synthesized_data_dir = args.synthesized_data_dir
    model_name = args.model_name
    dataset_name = args.dataset_name
    reference_data_dir = args.reference_data_dir
    metrics_list = args.metrics_list

    metrics_cal = MetricCalculator(
        synthetized_data_dir=synthesized_data_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        reference_data_dir=reference_data_dir
    )

    if metrics_list is not None:
        metrics = metrics_cal.get_metrics_by_list(metric_list=metrics_list)
    else:
        metrics = metrics_cal.get_all_metrics()

    print(f"Computed Metrics: {json.dumps(metrics, indent=4)}")

    save_dir = "metrics_results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"metrics_{model_name}_eval_for_dataset_{dataset_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(save_dir, f'metrics_{model_name}_eval_for_dataset_{dataset_name}.json')}")