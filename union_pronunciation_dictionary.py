import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_1", type=str, default="resources/filtered_cmu_dictionary.txt")
    parser.add_argument("--file_2", type=str, default="lexicon/librispeech-lexicon.txt")
    parser.add_argument("--save_dir", type=str, default="lexicon/")

    args = parser.parse_args()

    return args


def read_file_to_set(file_path):
    with open(file_path, "r", encoding="latin-1") as file:
        return set(line.strip() for line in file)


def find_union_of_lines(file1_set, file2_set):
    return file1_set.union(file2_set)


def write_lines_to_file(lines, output_file):
    with open(output_file, "w", encoding="latin-1") as file:
        for line in sorted(lines):
            file.write(line + "\n")


if __name__ == "__main__":
    args = get_args()

    file_1 = args.file_1
    file_2 = args.file_2
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    file1_set = read_file_to_set(file_path=file_1)
    file2_set = read_file_to_set(file_path=file_2)

    union_lines = find_union_of_lines(file1_set=file1_set, file2_set=file2_set)

    write_lines_to_file(lines=union_lines, output_file=os.path.join(save_dir, "libritts_lexicon_union_cmu_dict.txt"))