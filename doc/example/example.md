# Example

First of all, open a terminal, activate the virtual environment with the CoCoScore dependencies and
change to the directory of this example:

```bash
source activate cocoscore
cd doc/example
```

## Using a pre-trained scoring model

### Scoring sentences

The example dataset consists of sentences containing co-occurrences of genes and diseases. 

To extract sentences to be processed, execute in a terminal:

```bash
dataset_path=demo.tsv
sentences_path=sentences.txt
cut -f 6 "$dataset_path" > "$sentences_path"
```

We use a pre-trained fastText model to predict the probability that each sentence describes an association.
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

### Hyperparameter optimization via cross-validation

### Evaluation of test set performance

### Model compression

