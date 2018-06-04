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
- text co-mentioning the entities (string with names of entities blanked as described below). Please make sure that the text column does not contain tab characters (`\t`) as this will make the text appear to span multiple columns in the input file. This can be fixed by replacing tabs with spaces.

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
MYGENETOKEN mutations in patients with MYDISEASETOKEN.
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
Execute the following piece of code in Python.
fastText will be called in the process and write progress information to the command line.

```python
import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tools.data_tools as dt
import os

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
                   
os.remove(prob_path) # remove intermediary class probabilities file
```

### Computing co-occurrence scores

Based on the previously computed sentence scores, we can now compute the final disease-gene co-occurrence scores which are written to `co_occurrence_scores.tsv`
Please execute the following in Python:

```python
import cocoscore.tagger.co_occurrence_score as cos
import os

scored_dataset_path = 'demo_scored.tsv'
scores_path = 'co_occurrence_scores.tsv'

# only the first mandatory argument sentence_score_file_path of co_occurrence_score() matters here
cocoscores = cos.co_occurrence_score(score_file_path=scored_dataset_path,
                                     first_type=9606, second_type=-26,
                                     matches_file_path=None, entities_file=None)
with open(scores_path, 'wt') as fout:
    for pair, score in sorted(cocoscores.items(), key=lambda x: x[0]):
        fout.write('\t'.join(pair) + '\t' + str(score) + os.linesep)
```

## Advanced use case: Training and applying a custom scoring model to your own dataset

We now describe how you can train your own fastText model to score sentence-level co-occurrences. This step is necessary if other co-mentions than disease-gene co-mentions are to be scored or if you prefer to the model on your own corpus.
Please note that this section assumes a basic understanding of machine learning approaches such as the motivation behind training-test set splitting and cross-validation.

### Label column

When training a custom model, an additional column is needed in the dataset file that indicates whether the sentence is classified as positive or negative.
These class labels are to be specified as 1 (for positives) or 0 (for negatives).
For the purpose of this tutorial, we randomly assign each instance in `demo.tsv` to one of the classes and append the class labels to each line.
In a real world setting, we assign these labels using [distant supervision](#appendix-distant-supervision) or
you can use a manually labelled dataset, if available.
This is achieved by executing the following in a terminal:

```bash
awk 'BEGIN{srand(42);}{print $0"\t"int(2 * rand())}' demo.tsv > demo_labels.tsv
```

Before training the fastText model, we extract class labels (prefixed by `__label__` as required by fastText) and (lowercase) text by executing the following in a terminal:

```bash
awk -F '\t' '{print "__label__"$7" "tolower($6)}' demo_labels.tsv > sentences_labels.txt
```

### Fitting a model with fixed hyperparameters

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

This trains a fastText model using the given hyperparameter settings.
The final model is written to `mymodel.ftz` and fastText's progress is written to the command line.
The ending `.ftz` indicates that the model has been compressed using the`fasttext quantize` command.

### Splitting into training and test data

The next section will explain how the best hyperparameter settings can be found for a given dataset.
Before proceeding, please split your dataset into independent training and test sets by reserving, for instance, 20% of the association for the test set. We recommend that you perform the splitting by assigning **all** sentences co-mentioning a given pair either to training or test set, never both, to avoid underestimating the generalization error of the model.
This cross-validation strategy is implemented in the utility function
`cocoscore.ml.fasttext_helpers.fasttext_cv_independent_associations()` used in the example below.

While the training set is used to pick the optimal hyperparameters for the fastText model (see next section), sentence-level scores and co-occurrences scores for the test dataset can be used to assess the performance of the overall model

### Hyperparameter optimization via cross-validation

fastText comes with a number of important hyperparameters such as the number of training epochs,
learning rate or n-gram length.
While the example above uses a selection of hyperparameters that often yield good performance,
these hyperparameters should ideally be tuned for each dataset.
CoCoScore offers a set of functions to perform a random search cross-validation 
akin to the [strategy implemented in scikit-learn](http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization).

Such a cross-validation on the `demo_labels.tsv` dataset can be performed as follows:

```python
import numpy as np

import cocoscore.ml.cv as cv
import cocoscore.ml.fasttext_helpers as fth
import cocoscore.tools.data_tools as dt

data_path = 'demo_labels.tsv'
output_path = 'cv_results.tsv'

ft_path = 'fasttext'
dim = 300  # for this example, fix the dimensionality of the generated word embeddings to 300
cv_folds = 3  # for 3-fold cross-validation
cv_iterations = 5  # try out 5 randomly selected hyperparameters settings in cross-validation
ft_threads = 1  # the number of threads to use by fastText

data_df = dt.load_data_frame(data_path, sort_reindex=True)
data_df['text'] = data_df['text'].apply(lambda s: s.strip().lower())

def cv_function(input_df, params, random_state):
    return fth.fasttext_cv_independent_associations(input_df, params,
                                                    ft_path, cv_folds=cv_folds,
                                                    random_state=random_state,
                                                    thread=ft_threads)

cv_results = cv.random_cv(data_df, cv_function, cv_iterations, {'-dim': dim},
                          fth.get_hyperparameter_distributions(np.random.randint(1000)), 3)
with open(output_path, 'wt') as fout:
    cv_results.to_csv(fout, sep='\t')
```

This will produce temporary cross-validation datasets, compute train and test error (i.e. validation error) on each fold and report these errors for different parameter choices.
Note that the dimensionality of the word embeddings is fixed to 300 in the example above (the `dim` variable).
This means that the `-dim` parameter of fastText will not be subjected to cross-validation.
Fixing parameters like this allows to incorporate prior knowledge into the model training.

The outputs of the cross-validation are written to the file `cv_results.tsv` in a tab-separated format which may look like this:

| dim | epoch | lr     | wordNgrams | ws | mean_test_score    | ... |
|-----|-------|--------|------------|----|--------------------| ----|
| 300 | 32    | 5.44   | 5          | 7  | 0.5536735170844427 | ... |
| 300 | 19    | 0.0324 | 5          | 4  | 0.5241161763750893 | ... |
| 300 | 14    | 0.989  | 3          | 5  | 0.5537584015195791 | ... |
| 300 | 41    | 0.226  | 4          | 3  | 0.5622071678085158 | ... |
| 300 | 38    | 6.37   | 2          | 8  | 0.5092082278070005 | ... |


The model with the best cross-validation performance in the one with highest `mean_test_score` which can be found in row 4 (as expected, all hyperparameter settings perform poorly due to the random labelling).

This means that the following hyperparameter settings should be selected for the given dataset:

```python
hyperparams = {'-dim': 300, '-epoch': 41, '-lr': 0.226, '-wordNgrams': 4, '-ws': 4}
```

Note that in a real world setting the `cv_iterations` variable above should be greater than 5 to try out more hyperparameter combinations.

### Computing co-occurrence scores

To compute co-occurrence scores using your own model and hyperparameter settings, simply follow the steps outlined in the section 'Using a pre-trained scoring model' while replacing the pre-trained model `demo.ftz` with your own model `mymodel.ftz` when scoring sentences.

## Appendix: distant supervision

Distant supervision is an approach to label a large amount of data without manually inspecting each element
in the dataset to assign a label.
The approach is based on a curated *knowledge base* that contains known associations of interest.
 
In this example, the knowledge base consists of two disease-gene associations:

| disease             | gene  |
| ------------------- | ----- |
| Okihiro syndrome    | SALL4 |
| Parkinson's disease | PINK1 |

We furthermore have a set of unlabelled sentences co-mentioning diseases and genes:

> Okihiro syndrome is caused by SALL4 mutations.
> 
> Potential role of SALL4 in the development of Okihiro syndrome.
>
> SALL4: a new marker for Parkinson's disease?
>
> PINK1 is not associated with Parkinson's disease.
>
> PINK1 is linked to Crohn's disease.

In distant supervision, every sentence that co-mentions a disease-gene pair that is in our knowledge base
is assumed to be a positive example.
Every sentence where both disease and gene appear in the knowledge base but *not* their association is assumed to be
a negative example.
Sentences where either disease or gene do not appear in the knowledge base are removed from the dataset.

This leaves us with the following labelled set of sentences:

| sentence                                                        | label  |
| --------------------------------------------------------------- | ------ |
| Okihiro syndrome is caused by SALL4 mutations.                  | 1      |
| Potential role of SALL4 in the development of Okihiro syndrome. | 1      |
| SALL4: a new marker for Parkinson's disease?                    | 0      |
| PINK1 is not associated with Parkinson's disease.               | 1      |

Note that distant supervision results in a *noisy labelling*.
For instance, sentence four clearly describes the absence of an association but is still marked as a positive example because
the association Parkinson's disease-PINK1 appears in the knowledge base.
Applying distant supervision will usually result in a noisy but a much larger dataset than we could label by hand.
In our experience, this excess of training data allows the distantly supervised model to pick up subtle differences
between positive and negative examples, despite the noise in the labels.
