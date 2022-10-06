import os
import sys
import json
import csv
import gzip
import itertools
from collections import defaultdict, Counter
import numpy as np
import spacy
from typing import Dict, List
import random
import re
import argparse

from sklearn.model_selection import KFold


FIGURE_REGEX = r'^((Fig|FIG)(ure|URE)?\s?\.?\s?\d+(:|\.)\s?)'


def flatten(l: List) -> List:
    # flatten list of lists
    return [item for sublist in l for item in sublist]


def _not_valid(entry: Dict) -> bool:
    """
    Check if entry is valid
    :param entry:
    :return:
    """
    if not entry['annotated']:
        return True
    if all([-1 in lvl for lvl in entry['levels']]):
        return True
    if not entry['alt_text'].strip():
        return True
    return False


def _clean_sent(s: str) -> str:
    """
    Strip Figure X: from beginning of alt-text sentences
    :param s:
    :return:
    """
    return re.sub(FIGURE_REGEX, '', s).strip()


DATA_FILE = 'data/annotation_data.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to annotation data JSON file")
    parser.add_argument("--outdir", type=str, help="Output directory for data")
    args = parser.parse_args()

    DATA_FILE = args.data
    if not DATA_FILE or not(os.path.exists(DATA_FILE)):
        print('Data file does not exist!')
        sys.exit(-1)

    TRAINING_DATA_DIR = args.outdir
    if not TRAINING_DATA_DIR:
        print('Invalid output directory')
        sys.exit(-1)
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    nlp = spacy.load("en_core_sci_sm")

    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f'Entries: {len(data)}')

    training_data_by_doc = []
    for entry in data:
        if _not_valid(entry):
            continue
        training_data = []
        if len(entry['sentences']) != len(entry['levels']):
            print('Warning: different numbers of annotations and sentences!')
            print(f"\t{entry['sentences']}")
            print(f"\t{entry['levels']}")
        for sent_ind, (sent, lvl) in enumerate(zip(entry['sentences'], entry['levels'])):
            lvl_array = [0, 0, 0, 0]
            for l in lvl:
                if l > 0:
                    lvl_array[l-1] = 1
            training_data.append({
                "corpus_id": entry['corpus_id'],
                "sent_id": sent_ind,
                "text": _clean_sent(sent),
                "labels": lvl_array
            })
        training_data_by_doc.append(training_data)

    print(f'{len(training_data_by_doc)} valid training entries.')

    # write output
    all_output_file = os.path.join(TRAINING_DATA_DIR, f'all_data.jsonl')
    flat_data = flatten(training_data_by_doc)
    with open(all_output_file, 'w') as outf:
        for entry in flat_data:
            json.dump(entry, outf)
            outf.write('\n')

    # split into folds
    num_folds = 5
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=12345)

    # create splits for cross-validation and write to file
    for i, (train_inds, val_inds) in enumerate(cv.split(training_data_by_doc)):
        print(f'Fold {i}')

        output_subdir = os.path.join(TRAINING_DATA_DIR, f'{i:02d}')
        os.makedirs(output_subdir, exist_ok=True)

        train_data = [training_data_by_doc[ind] for ind in train_inds]
        val_data = [training_data_by_doc[ind] for ind in val_inds]
        train_flat = flatten(train_data)
        val_flat = flatten(val_data)

        random.shuffle(train_flat)
        random.shuffle(val_flat)

        train_outfile = os.path.join(output_subdir, 'train.jsonl')
        val_outfile = os.path.join(output_subdir, 'val.jsonl')

        with open(train_outfile, 'w') as outf:
            for entry in train_flat:
                json.dump(entry, outf)
                outf.write('\n')

        with open(val_outfile, 'w') as outf:
            for entry in val_flat:
                json.dump(entry, outf)
                outf.write('\n')

    print('done.')