# HCI Alt Text Dataset

This repository contains the dataset and analysis scripts for the project described in the 2022 ASSETS paper:

* [Sanjana Chintalapati, Jonathan Bragg, Lucy Lu Wang. "A Dataset of Alt Texts from HCI Publications: Analyses and Uses Towards Producing More Descriptive Alt Texts of Data Visualizations in Scientific Papers." ASSETS 2022.](https://arxiv.org/abs/2209.13718)

We extract a dataset of alt text from HCI publications from the last decade, and provide annotations of the types of semantic content included in these alt texts based on the framework introduced by [Lundgard and Satyanarayan](http://vis.csail.mit.edu/pubs/vis-text-model/). Most figures in scientific papers lack alt text and we hope this dataset can help to improve the situation.

## Dataset Access

The dataset is contained in `data/hci-alt-text-dataset-20220915.jsonl`. This is a jsonlines file where each line contains one alt text along with its annotations.

An example:

```python
{
  'title': 'Effect of target size on non-visual text-entry',
  'pdf_hash': 'f65de96c15e33484bc85079b354f053fd9fe8ccc',
  'year': 2016,
  'venue': 'MobileHCI',
  'alt_text': 'The graph shows a significant increase of the relative path lenght as size gets smaller. From a relative distance smaller than 10 pixels to a 30 pixel difference on tiny. The same effect it is also noticible on the task axis lengh but the difference between tiny and large is of only about 8 relative pixels.',
  'levels': [[3], [3, 2], [3, 2]],
  'corpus_id': 17354027,
  'sentences': [
    'The graph shows a significant increase of the relative path lenght as size gets smaller.',
    'From a relative distance smaller than 10 pixels to a 30 pixel difference on tiny.',
    'The same effect it is also noticible on the task axis lengh but the difference between tiny and large is of only about 8 relative pixels.'
  ],
  'caption': 'Figure 3- Relative Path Length and Task Axis Length in pixels.',
  'local_uri': ['f65de96c15e33484bc85079b354f053fd9fe8ccc_Image_005.jpg'],
  'annotated': True,
  'is_plot': True,
  'uniq_levels': [2, 3],
  'has_1_2_3': False,
  'has_1_2': False,
  'compound': False
}
 ```

The `alt_text` field contains the original extracted author-written alt text. The `sentences` field contains the sentence splits for the alt text, and `levels` the annotated semantic levels corresponding to each sentence.

Not all instances in this data file have been annotated, since some images correspond to natural images, schematics, and other types of diagrams which fall outside of the scope of our annotation. To filter down to only annotated alt text, select those entries where the `annotated` flag is True.

There is also a field called `compound`. Some figures are compound, consisting of multiple images extracted from the PDF. These figures have been labeled as such. 

For access to the images associated with the dataset, download and unzip [this file](https://ai2-s2-hci-alt-texts.s3-us-west-2.amazonaws.com/images.tar.gz) (`size: 82Mb, md5: f6ff2c06, sha1: 781a5815`) into a local directory. The URIs in the dataset point to the corresponding files.

## Setup environment

1. Clone this repo into a directory such as `~/git/hci-alt-texts/`
2. Install Miniconda following the instructions [here](https://docs.conda.io/en/latest/miniconda.html)
3. From the repo home, run `conda env create -n hci-alt-texts -f environment.yml`
4. Run `conda activate hci-alt-texts`
5. Now you should be able to execute all of the analysis scripts. You may be prompted in some cases to install additional libraries or datasets.

## Analysis

Analysis scripts and scripts used to generate plots for the paper are available in the jupyter notebook `notebooks/hci-alt-text-analysis.ipynb`. Please set up your environment before running.

## Semantic level classifiers

To prepare the dataset for training, use `classifier/semantic_level_classifier/prep_data.py`. This script will split the dataset into five splits for cross-fold validation.

Example usage:
```bash
python models/semantic_level_classifier/prep_data.py --data data/hci-alt-text-dataset-20220915.jsonl --outdir models/semantic_level_classifier/new_data_splits/
```

The data splits used to train and evaluate the BERT and SciBERT classifiers can be found in `models/semantic_level_classifier/data_splits/`

### Random forest classifier

The training script for the random forest model can be found at `models/semantic_level_classifier/rf_classifier.py`

Example usage:
```bash
python models/semantic_level_classifier/rf_classifier.py --data models/semantic_level_classifier/data_splits/all_data.jsonl --folds 5 --option tfidf --outdir models/semantic_level_classifier/rf_model/
```

Two vectorization options can be specified: `tfidf` or `spacy`.

### BERT and SciBERT classifiers

The training script for the BERT and SciBERT classifiers can be found at `models/semantic_level_classifier/scibert_classifier.py`

Example usage:
```bash
python models/semantic_level_classifier/scibert_classifier.py --model allenai/scibert_scivocab_uncased --data models/semantic_level_classifier/data_splits/ --outdir models/semantic_level_classifier/
```

The model variants trained for the paper start from base models: `bert-base-uncased` and `allenai/scibert_scivocab_uncased`.

The data input should point to the directory containing data splits generated using `prep_data.py`.

## Citation

If using this dataset, please cite:

```
@inproceedings{Chintalapati2022ADO,
  title={A Dataset of Alt Texts from {HCI} Publications: Analyses and Uses Towards Producing More Descriptive Alt Texts of Data Visualizations in Scientific Papers},
  author={Sanjana Shivani Chintalapati and Jonathan Bragg and Lucy Lu Wang},
  booktitle="Proceedings of the 24th International ACM SIGACCESS Conference on Computers and Accessibility",
  month=oct,
  year={2022},
  publisher="Association for Computing Machinery",
  doi="10.1145/3517428.3544796"
}
```
