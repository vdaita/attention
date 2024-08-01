from datasets import load_dataset, Dataset
import difflib
import diff_utils
from tqdm import tqdm

ds = load_dataset("nuprl/EditPackFT", split="train")
# Generate a diff, mask some lines of code arbitrarily, and train on predicting the commit message
ds = ds.select(range(10))
generated_dict = {
    "before": [],
    "intermediate": [],
    "after": [],
    "subject": [],
    "intermediate_diff": []
}
for row in tqdm(ds):
    before_lines = row['old_contents'].splitlines()
    after_lines = row['new_contents'].splitlines()

    diff = list(difflib.unified_diff(before_lines, after_lines, n=100000000))
    new_lines = []

    largest_contiguous_add_length = 0
    largest_contiguous_add_start_line = 0

    line_idx = 0
    while line_idx < len(diff):
        if diff[line_idx].startswith("+"):
            contiguous_add = 0
            contiguous_add_start = line_idx
            while line_idx < len(diff) and diff[line_idx].startswith("+") :
                line_idx += 1
                contiguous_add += 1
            if contiguous_add > largest_contiguous_add_length:
                largest_contiguous_add_length = contiguous_add
                largest_contiguous_add_start_line = contiguous_add_start
        else:
            line_idx += 1

    if largest_contiguous_add_length > 6:
        continue

    # intermediate_diff = diff
    intermediate_diff = diff[:largest_contiguous_add_start_line + 3] + diff[largest_contiguous_add_start_line + largest_contiguous_add_length:]
    diff_sr = diff_utils.parse_diff("\n".join(intermediate_diff))
    partial_edited = row['old_contents']
    for sr in diff_sr:
        partial_edited = partial_edited.replace(sr.search_block, sr.replace_block)
    
    generated_dict["before"].append(row['old_contents'])
    generated_dict["intermediate"].append(''.join(difflib.restore(intermediate_diff, 2)))
    generated_dict["after"].append(row['new_contents'])
    generated_dict["subject"].append(row['subject'])
    generated_dict["intermediate_diff"].append("\n".join(intermediate_diff))

    print(generated_dict["before"][-1])
    print("-----------")
    print(generated_dict["intermediate"][-1])
    print("-----------")
    print(generated_dict["after"][-1])
    print("-----------")
    print(generated_dict["intermediate_diff"][-1])
    print("===========")


diff_ds = Dataset.from_dict(generated_dict)
# diff_ds.push_to_hub("vdaita/EditPackFTIntermediate")