# Example - computing context-aware co-occurrence scores with CoCoScore

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

This section explains how context-aware co-occurrence scores can be computed using a pre-trained fastText model. The model `demo.ftz` has been trained to score co-occurrences of diseases and genes.

### Scoring sentences

To extract (and lowercase) sentences to be processed from the example dataset `demo.tsv`, execute in a terminal:

```bash
dataset_path=demo.tsv
sentences_path=sentences.txt
cut -f 6 "$dataset_path" | awk '{print tolower($0);}' > "$sentences_path"
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

## Training and applying a custom scoring model

We now describe how you can train your own model to score sentence-level co-occurrences. This step is necessary if other co-mentions than disease-gene co-mentions are to be scored.

### Label column

When training a custom model, an additional column is needed in the dataset file that indicates whether the sentence is classified as positive or negative.
These class labels are to be specified as 1 (for positives) or 0 (for negatives).
For the purpose of this tutorial, we randomly assign each instance in `demo.tsv` to one of the classes and append the class labels to each line.
This is achieved by executing the following in a terminal:

```bash
awk 'BEGIN{srand(42);}{print $0"\t"int(2 * rand())}' demo.tsv > demo_labels.tsv
```

Before training the fastText model, we extract class labels (prefixed by `__label__` as required by fastText) and (lowercase) text by executing the following in a terminal:

```bash
awk -F '\t' '{print "__label__"$7" "tolower($6)}' demo_labels.tsv > sentences_labels.txt
```

### Fitting a model with fixed parameters

To train the model, execute the following in Python:

```python
from cocoscore.ml.fasttext_helpers import fasttext_fit

train_path = 'sentences_labels.txt'
params = {'-dim': 300, '-epoch': 10, '-lr': 0.01}
fasttext_path = 'fasttext'

model_file = fasttext_fit(train_path, params, fasttext_path, thread=1, compress_model=True,
                                  model_path='mymodel')
print(model_file)
# mymodel.ftz
```

This trains a fastText model using the given parameter settings.
The final model is written to `mymodel.ftz`.
The ending `.ftz` indicates that the model has been compressed using the`fasttext quantize` command.

### Splitting into training and test data

In practice, we make sure that training and test set are independent by entity pairs are either exclusively used for training or testing, never both.

### Evaluation of training and test set performance

### Hyperparameter optimization via cross-validation

### Computing co-occurrence scores

To compute co-occurrence scores using your own model, simply follow the steps outlined in the section 'Using a pre-trained scoring model' while replacing the pre-trained model `demo.ftz` with your own model `mymodel.ftz` when scoring sentences.
