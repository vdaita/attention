from datasets import load_dataset, Dataset
import difflib
import diff_utils
from tqdm import tqdm

ds = load_dataset("nuprl/EditPackFT", split="train")
# Generate a diff, mask some lines of code arbitrarily, and train on predicting the commit message

generated_dict = {
    "before": [],
    "intermediate_changes": [],
    "after": [],
    "subject": [],
    "intermediate_diff": []
}
for row in tqdm(ds):
    before_lines = row['old_contents'].splitlines()
    after_lines = row['new_contents'].splitlines()
    
    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    lines_written = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            lines_written += j2 - j1
        elif tag == 'insert':
            lines_written += j2 - j1
    if lines_written < 10:
        continue

    intermediate_change_file = ""
    
    lines_count = 0
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            intermediate_change_file += "\n".join(before_lines[i1:i2]) + "\n"
        elif tag == 'insert': # make sure that this isn't an import statement
            if lines_count < 2:
                for line in after_lines:
                    if not("from" in line or "import" in line):
                        intermediate_change_file += line + "\n"
                        lines_count += 1
                # intermediate_change_file += "\n".join(after_lines[j1:j2])
        elif tag == 'replace':
            if lines_count < 2:
                for line in after_lines:
                    if not("from" in line or "import" in line):
                        intermediate_change_file += line + "\n"
                        lines_count += 1
            else:
                intermediate_change_file += "\n".join(before_lines[i1:i2])
        elif tag == 'delete':
            intermediate_change_file += "\n".join(before_lines[i1:i2])

    intermediate_diff = "\n".join(difflib.unified_diff(before_lines, intermediate_change_file.splitlines(), n=2))
    
    generated_dict["before"].append(row['old_contents'])
    generated_dict["after"].append(row['new_contents'])
    generated_dict["intermediate_changes"].append(intermediate_change_file)
    generated_dict["subject"].append(row['subject'])
    generated_dict["intermediate_diff"].append(intermediate_diff)

diff_ds = Dataset.from_dict(generated_dict)
diff_ds.push_to_hub("vdaita/EditPackFTIntermediate")