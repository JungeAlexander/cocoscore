![CoCoScore-text-small.png](doc/logos/CoCoScore-text-small.png) 

# Context-aware co-occurrence scores for text mining applications

[![Build Status](https://travis-ci.org/JungeAlexander/cocoscore.svg?branch=master)](https://travis-ci.org/JungeAlexander/cocoscore)
 
 
Text mining of the biomedical literature has been successful in retrieving interactions between proteins, non-coding RNAs, and chemicals as well as in determining tissue-specific expression and subcellular localization. Simple co-occurrence-based scoring schemes can uncover such associations by finding entity pairs that are frequently mentioned together but ignore the textual context of each co-occurrence.

CoCoScore implements an improved context-aware co-occurrence scoring scheme that uses textual context to assess whether an association is described in a given sentence or not. CoCoScore achieves superior performance compared to previous approaches that rely on constant sentence scores, based on datasets of disease-gene, tissue-gene, and protein-protein associations. 
In our research, we use [distant supervision](doc/example/example.md#appendix-distant-supervision) to create an automatic, but noisy, labelling of a large dataset of sentences co-mentioning two entities of interest.

## Quick start

1. Follow the [installation instructions](#installation).
2. Download the [files](#example-usage) needed to run through the example.
3. Run through the [example](doc/example/example.md) to learn how to apply CoCoScore to your own data.
 
## Installation

To install CoCoScore, first clone the repository and add the directory to your PYTHONPATH, e.g.:

```bash
git clone git@github.com:JungeAlexander/cocoscore.git
cd cocoscore
export PYTHONPATH="`pwd`:$PYTHONPATH"
```

CoCoScore has been tested on Linux and Mac OS.
We recommend installing all dependencies in a virtual environment as described in the next section as this is the more convenient way.

CoCoScore is written in Python and depends on a range of Python packages such as scikit-learn, nltk and gensim.
Furthermore [fastText](https://github.com/facebookresearch/fastText) is a dependency.

If you decide not to setup a virtual environment using conda, please build fastText as described [here](https://github.com/facebookresearch/fastText#building-fasttext) and make sure the `fasttext` binary is discoverable via your `$PATH` environment variable.
The complete list of dependencies and version numbers CoCoScore has been tested with can be found in [environment.yml](environment.yml).


### Create a virtual environment using conda

Please install the Python 3.6 version of either miniconda or anaconda as described here:
https://conda.io/docs/user-guide/install/download.html

Afterwards, prepare a virtual environment for CoCoScore by executing the following in a terminal:

```bash
# install all dependencies in virtual environment called cocoscore
conda env create -f environment.yml

# active virtual environment that was just created
source activate cocoscore
```

## Example usage

Before running the examples, please download the following files:

- [Pre-trained fastText model](http://download.jensenlab.org/BLAH4/demo.ftz)
- [Training set](http://download.jensenlab.org/BLAH4/medlinepmcoa_train_ghr.tsv.gz) of sentences co-mentioning diseases and genes derived from Medline abstracts and PubMed Central Open Access articles
- [Test set](http://download.jensenlab.org/BLAH4/medlinepmcoa_test_ghr.tsv.gz) corresponding to the training set above

The files are downloaded and placed in the correct directories by executing:

```shell
wget -P doc/example/ http://download.jensenlab.org/BLAH4/demo.ftz
wget http://download.jensenlab.org/BLAH4/medlinepmcoa_train_ghr.tsv.gz http://download.jensenlab.org/BLAH4/medlinepmcoa_test_ghr.tsv.gz
```

Afterwards, please see [here](doc/example/example.md) for usage examples.

## Contributors

CoCoScore is being developed by Alexander Junge and Lars Juhl Jensen at the
Disease Systems Biology Program, Novo Nordisk Foundation Center for Protein Research,
Faculty of Health and Medical Sciences, University of Copenhagen, Denmark.

## Feedback

Please open an issue here or write us:
`{alexander.junge,lars.juhl.jensen} AT cpr DOT ku DOT dk`
