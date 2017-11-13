# Example

First of all, open a terminal, activate the virtual environment with the CoCoScore dependencies and
change to the directory of this example:

```bash
source activate cocoscore
cd doc/example
```

## Data preparation

The dataset is expected to come as a tab-delimited file (no header) with the following columns:

- document identifier (integer, e.g. PubMed ID)
- paragaph number (integer)
- sentence number (integer)
- first co-mentioned entity (string, e.g. disease identifier)
- second co-mentioned entity (string, e.g. gene identifier)
- text co-mentioning the entities (string with names of entities blanked as described below)

See `demo.tsv` for an example of a file in the correct format.
The file contains sentence-level co-mentions of hemochromatosis (iron storage disorder) and different genes that were extracted from Medline abstracts and PMC Open Access articles.

### Blanking named entities in text

Names of the co-mentioned entities in the dataset's text column must be blanked.
Only this allows the scoring model to generalize to unseen entitiy pairs instead of overfitting to examples in the dataset used for training.

We recommend replacing entity names with a fixed placeholder that reflects their type.
For instance, gene (TFR2) and disease (hemochromatosis) in the sentence

```
TFR2 mutations in patients with hemochromatosis.
```

could be replaced by tokens `MYGENETOKEN` and `MYDISEASETOKEN`, respectively:

```
MYGENETOKEN mutations inpatients with MYDISEASETOKEN.
```

This strategy is followed in our `demo.tsv` example dataset.

## Using a pre-trained scoring model

This section explains how context-aware co-occurrence scores can be computed using a pre-trained fastText model.

### Scoring sentences

To extract sentences to be processed from the example dataset `demo.tsv`, execute in a terminal:

```bash
dataset_path=demo.tsv
sentences_path=sentences.txt
cut -f 6 "$dataset_path" > "$sentences_path"
```

We use the previously downloaded pre-trained fastText model `demo.ftz` to predict the probability that each sentence describes an association.
The sentence scores are then written to the file `demo_scored.tsv`.
Execute the following in Python:

```python
import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tools.data_tools as dt

model_path = 'demo.ftz'
dataset_path = 'demo.tsv'
sentences_path = 'sentences.txt'
fasttext_path = 'fasttext'
prob_path = 'probabilities.txt.gz'
scored_dataset_path = 'demo_scored.tsv'

fth.fasttext_predict(model_path, sentences_path, fasttext_path, prob_path)
probabilities = fth.load_fasttext_class_probabilities(prob_path)

df = dt.load_data_frame(dataset_path, class_labels=False)
df['predicted'] = probabilities
with open(scored_dataset_path, 'wt') as test_out:
    df.to_csv(test_out, sep='\t', header=False, index=False,
                   columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])
```

### Computing co-occurrence scores

Based on the previously computed sentence scores, we can now compute the final disease-gene co-occurrence scores which are written to `co_occurrence_scores.tsv`
Please execute the following in Python:

```python
import cocoscore.tagger.co_occurrence_score as cos
import os

scores_path = 'co_occurrence_scores.tsv'

cocoscores = cos.co_occurrence_score(score_file_path=scored_dataset_path,
                                      matches_file_path=None, entities_file=None)
with open(scores_path, 'wt') as fout:
    for pair, score in sorted(cocoscores.items(), key=lambda x: x[0]):
        fout.write('\t'.join(pair) + '\t' + str(score) + os.linesep)
```

## Training and testing your own scoring model

### Training and test datasets

### Evaluation of test set performance

### Model compression

