
![CoCoScore-text-small.png](doc/logos/CoCoScore-text-small.png) 

# Context-aware co-occurrence scores for text mining applications

[![Build Status](https://travis-ci.org/JungeAlexander/cocoscore.svg?branch=master)](https://travis-ci.org/JungeAlexander/cocoscore)
 

TODO 

## Performance

TODO

## Dependencies

CoCoScore has been tested on Linux and Mac OS.
We recommend installing all dependencies in a virtual environment as described in the next section.

CoCoScore is written in Python and depends on a range of Python packages such as scikit-learn, nltk and gensim.
Furthermore [fastText](https://github.com/facebookresearch/fastText) is a dependency.
The complete list of dependencies and version numbers CoCoScore has been tested with can be found in [environment.yml](environment.yml).

### Create a virtual environment using conda

Please install the Python 3.6 version of either miniconda or anacoda as described here:
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
