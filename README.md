![CoCoScore-text-small.png](doc/logos/CoCoScore-text-small.png) 

# Context-aware co-occurrence scores for text mining applications

[![Build Status](https://travis-ci.org/JungeAlexander/cocoscore.svg?branch=master)](https://travis-ci.org/JungeAlexander/cocoscore)
 
 
Text mining of the biomedical literature has been successful in retrieving interactions between proteins, non-coding RNAs, and chemicals as well as in determining tissue-specific expression and subcellular localization. Simple co-occurrence-based scoring schemes can uncover such associations by finding entity pairs that are frequently mentioned together but ignore the textual context of each co-occurrence.

CoCoScore implements an improved context-aware co-occurrence scoring scheme that uses textual context to assess whether an association is described or not. CoCoScore achieves an area under the ROC curve of 0.94, compared to 0.92 for previous approaches, based on a dataset of curated disease-gene associations. 
 
## Installation

To install CoCoScore, first clone the repository and add the directory to your PYTHONPATH, e.g.:

```bash
git clone git@github.com:JungeAlexander/cocoscore.git
cd cocoscore
export PYTHONPATH="`pwd`:$PYTHONPATH"
```

CoCoScore has been tested on Linux and Mac OS.
We recommend installing all dependencies in a virtual environment as described in the next section.

CoCoScore is written in Python and depends on a range of Python packages such as scikit-learn, nltk and gensim.
Furthermore [fastText](https://github.com/facebookresearch/fastText) is a dependency.
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

See [here](doc/example/example.md) for usage examples. 
