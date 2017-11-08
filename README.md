![CoCoScore-small.png](https://raw.githubusercontent.com/JungeAlexander/cocoscore/master/doc/logos/CoCoScore-small.png)

[![Build Status](https://travis-ci.org/JungeAlexander/cocoscore.svg?branch=master)](https://travis-ci.org/JungeAlexander/cocoscore)

# CoCoScore: context-aware co-occurrence scores for text mining applications

## Background

TODO

## Performance

TODO

## Dependencies

We recommend installing all dependencies in a virtual environment as described in the next section.

CoCoScore depends on a range of Python packages such as scikit-learn, nltk and gensim.
Furthermore [fastText](https://github.com/facebookresearch/fastText) is a dependency.
The complete list of dependencies and version numbers CoCoScore has been tested with can be found in `environment.yml`.

### Create a virtual environment using conda

Please install the Python 3.6 version of either miniconda or anacoda as described here:
https://conda.io/docs/user-guide/install/download.html

Afterwards, prepare a virtual environment for CoCoScore by executing:

```bash
# install all dependencies in virtual environment called cocoscore
conda env create -f environment.yml

# active virtual environment that was just created
source activate cocoscore
```

## Example usage

See [here](doc/example/example.md) for usage examples. 
