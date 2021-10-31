import os
import sys

_, manifest, n_splits = sys.argv

with open(manifest) as inf:
    rows = inf.readlines()
    root, *wavs = rows
    n_splits = int(n_splits)
    split_len = int(len(rows) / n_splits) + 1
    start = 0
    end = split_len
    for i in range(1, n_splits + 1):
        base, ext = os.path.splitext(manifest)
        meta_out = f"{base}_{str(i)}{ext}"
        with open(meta_out, "w") as outf:
            outf.write(root)
            outf.writelines(wavs[start:end])
            start = end
            end += split_len
