#!/usr/bin/env python
# Note, updated version of
# https://github.com/ipython/ipython-in-depth/blob/master/tools/nbmerge.py
"""
usage:
python nbmerge.py A.ipynb B.ipynb C.ipynb > merged.ipynb
"""

import io
import os
import sys

import json

def merge_notebooks(filenames):
    merged = None
    for fname in filenames:
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        if merged is None:
            merged = nb
        else:
            # TODO: add an optional marker between joined notebooks
            # like an horizontal rule, for example, or some other arbitrary
            # (user specified) markdown cell)
            merged["cells"].append(nb["cells"])

    print(json.dumps(merged, indent=4))

if __name__ == '__main__':
    notebooks = sys.argv[1:]
    if not notebooks:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    merge_notebooks(notebooks)
