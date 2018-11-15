================================================================================
CoCoScore: context-aware co-occurrence scores for text mining applications
================================================================================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|



.. |travis| image:: https://travis-ci.org/JungeAlexander/cocoscore.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/JungeAlexander/cocoscore

.. |version| image:: https://img.shields.io/pypi/v/cocoscore.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/cocoscore

.. |commits-since| image:: https://img.shields.io/github/commits-since/JungeAlexander/cocoscore/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/JungeAlexander/cocoscore/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/cocoscore.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/cocoscore

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/cocoscore.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/cocoscore

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/cocoscore.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/cocoscore


.. end-badges

.. image:: https://github.com/JungeAlexander/cocoscore/blob/master/doc/logos/CoCoScore-text-small.png

Text mining of the biomedical literature has been successful in retrieving interactions between proteins, non-coding RNAs, and chemicals as well as in determining tissue-specific expression and subcellular localization. Simple co-occurrence-based scoring schemes can uncover such associations by finding entity pairs that are frequently mentioned together but ignore the textual context of each co-occurrence.

CoCoScore implements an improved context-aware co-occurrence scoring scheme that uses textual context to assess whether an association is described in a given sentence or not. CoCoScore achieves superior performance compared to previous approaches that rely on constant sentence scores, based on datasets of disease-gene, tissue-gene, and protein-protein associations.
In our research, we use `distant supervision <https://github.com/JungeAlexander/cocoscore/blob/master/doc/example/example.md#appendix-distant-supervision>`_ to create an automatic, but noisy, labelling of a large dataset of sentences co-mentioning two entities of interest.

Free software: MIT license


Installation
============

::

    pip install cocoscore


CoCoScore also depends on `fastText <https://fasttext.cc/>`_.
Please build v0.1.0 of fastText as described `here <https://github.com/facebookresearch/fastText/#building-fasttext-using-make-preferred>`_ and make sure the ``fasttext`` binary is discoverable via your ``$PATH`` environment variable.


fastText v0.1.0 is also available via `conda-forge <https://anaconda.org/conda-forge/fasttext>`_:


::

   conda install -c conda-forge fasttext=0.1.0


Quick start
===========

1. Follow the installation instructions above.
2. Download the ``demo.ftz`` file (see next section) needed to run through the example.
3. Run through the `example <https://github.com/JungeAlexander/cocoscore/blob/master/doc/example/example.md>`_ to learn how to apply CoCoScore to your own data.


Example usage
==============

Before running the examples, please download the following file and save it to ``doc/example/``:

- `Example pre-trained fastText model <http://download.jensenlab.org/BLAH4/demo.ftz>`_

The files are downloaded and placed in the correct directories by executing:

::

    wget -P doc/example/ http://download.jensenlab.org/BLAH4/demo.ftz


Preprint manuscript
====================

A preprint manuscript describing CoCoScore and its performance on eight datasets, compared to a baseline co-occurrence scoring model, is available `via bioRxiv <https://www.biorxiv.org/content/early/2018/10/16/444398>`_.

Supplementary data described in the manuscript can be downloaded `via figshare <https://doi.org/10.6084/m9.figshare.7198280.v1>`_.


Contributors
=============

CoCoScore is being developed by Alexander Junge and Lars Juhl Jensen at the
Disease Systems Biology Program, Novo Nordisk Foundation Center for Protein Research,
Faculty of Health and Medical Sciences, University of Copenhagen, Denmark.


Feedback
===========

Please open an issue here or write us:
``{alexander.junge,lars.juhl.jensen} AT cpr DOT ku DOT dk``

See also: https://github.com/JungeAlexander/cocoscore/blob/master/CONTRIBUTING.rst


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

