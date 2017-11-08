# Example

First of all, activate the virtual environment with the CoCoScore dependencies:

```bash
source activate cocoscore
```

## Using a pre-trained scoring model

TODO:

- explain gene/disease wildcards
- document what is going in here

```bash
cd doc/example
dataset_path=demo.tsv
model_path=demo.ftz
```

### Scoring sentences

```bash
sentences_path=demo.txt
cut -f 6 "$dataset_path" > "$sentences_path"
```

```python
import cocoscore.ml.fasttext_helpers as fth

model_path = 'demo.ftz'
sentences_path = 'demo.txt'
fasttext_path = 'fasttext'
prob_path = 'probabilities.txt.gz'


fth.fasttext_predict(model_path, sentences_path, fasttext_path, prob_path)
probabilities = fth.load_fasttext_class_probabilities(prob_path)

# TODO write scored sentences to file
    df = dt.load_data_frame(dataset_to_test_path[dataset], sort_reindex=True)
    df['predicted'] = scores
    with gzip.open(score_file_path, 'wt') as test_out:
        df.to_csv(test_out, sep='\t', header=False, index=False,
                       columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])
```

### Computing CoCoScores

```python
import cocoscore.tagger.co_occurrence_score as coco

scored_dataset_path = 'demo_scored.txt'
cocoscores = coco.co_occurrence_score(score_file_path=scored_dataset_path,
                                      matches_file_path=None,entities_file=None)
# TODO write to file
```

## Training and testing your own scoring model

TODO

### Training and test datasets

### Hyperparameter optimization via cross-validation

### Evaluation of test set performance

### Model compression

