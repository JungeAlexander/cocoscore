========
CoCoScore: context-aware co-occurrence scores for text mining applications
========

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

Text mining of the biomedical literature has been successful in retrieving interactions between proteins, non-coding RNAs, and chemicals as well as in determining tissue-specific expression and subcellular localization. Simple co-occurrence-based scoring schemes can uncover such associations by finding entity pairs that are frequently mentioned together but ignore the textual context of each co-occurrence.

CoCoScore implements an improved context-aware co-occurrence scoring scheme that uses textual context to assess whether an association is described in a given sentence or not. CoCoScore achieves superior performance compared to previous approaches that rely on constant sentence scores, based on datasets of disease-gene, tissue-gene, and protein-protein associations.
In our research, we use `distant supervision <https://github.com/JungeAlexander/cocoscore/blob/master/doc/example/example.md#appendix-distant-supervision>`_ to create an automatic, but noisy, labelling of a large dataset of sentences co-mentioning two entities of interest.

* Free software: MIT license

.. image:: https://github.com/JungeAlexander/cocoscore/blob/master/doc/logos/CoCoScore-text-small.png

Installation
============

::

    pip install cocoscore

Documentation
=============


To use the project:

.. code-block:: python

    import cocoscore
    cocoscore.longest()


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
