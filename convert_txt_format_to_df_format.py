import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--txt_dir", type=str, default="resources/filelists/ljspeech")

    args = parser.parse_args()

    return args

def parse_filelist(filelist_path, split_char="|"):
    result = {}
    result["file"] = []
    result["script"] = []
    with open(filelist_path) as f:
        lines = f.read()
        lines = lines.split("\n")
        lines = list(filter(None, lines))
        for line in lines:
            print(f"line: {line}")
            file, script = line.strip().split(split_char)
            result["file"].append(file)
            result["script"].append(str(script))
    return result

if __name__ == "__main__":
    args = get_args()
    txt_dir = args.txt_dir

    train_txt = os.path.join(txt_dir, "train.txt")
    valid_txt = os.path.join(txt_dir, "valid.txt")
    test_txt = os.path.join(txt_dir, "test.txt")

    train_dict = parse_filelist(filelist_path=train_txt)
    train_df = pd.DataFrame.from_dict(train_dict)
    train_df.to_csv(os.path.join(txt_dir, "train.csv"))

    valid_dict = parse_filelist(filelist_path=valid_txt)
    valid_df = pd.DataFrame.from_dict(valid_dict)
    valid_df.to_csv(os.path.join(txt_dir, "valid.csv"))

    test_dict = parse_filelist(filelist_path=test_txt)
    test_df = pd.DataFrame.from_dict(test_dict)
    test_df.to_csv(os.path.join(txt_dir, "test.csv"))