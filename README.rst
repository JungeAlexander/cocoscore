========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
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

CoCoScore: context-aware co-occurrence scores for text mining applications

* Free software: MIT license

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
