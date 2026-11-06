

Cite rgmin
----------

This page gives a BibTeX entry for rgmin that names the version you ran.

Get the entry
~~~~~~~~~~~~~

``CITATION.cff`` at the top of the repository is the citation record, in the Citation File Format (CFF).
cffconvert turns it into BibTeX.
Run it at the top of a checkout:

.. code:: bash

    pip install cffconvert
    cffconvert -f bibtex

On ``develop`` it prints:

::

    @misc{YourReferenceHere,
    author = {Goswami, Rohit},
    title = {rgmin: local minimization over eindir differentiable objectives},
    url = {https://github.com/OmniPotentRPC/rgmin}
    }

The "Cite this repository" button on GitHub reads the record on the default branch, ``main``.

Find the version you ran
~~~~~~~~~~~~~~~~~~~~~~~~

The record on ``develop`` names no version.
The records on ``main`` and at the ``v0.5.2`` tag say version 0.4.0 of 11 October 2026.
That is older than the release, so take the version from Cargo instead.
In the project that depends on rgmin, run:

.. code:: bash

    cargo tree -p rgmin --depth 0

With the release from crates.io, it prints:

::

    rgmin v0.5.2
    [build-dependencies]

The second line is a heading of ``cargo tree``, with nothing under it at depth 0.

With the ``develop`` branch, the short commit follows the ``#``:

::

    rgmin v0.2.0 (https://github.com/OmniPotentRPC/rgmin?branch=develop#06a41e5b)
    [build-dependencies]

On that route, cite the commit, since the ``Cargo.toml`` on ``develop`` still says 0.2.0.
:doc:`install` says why.
``Cargo.lock`` has the full commit:

.. code:: bash

    grep -A2 '^name = "rgmin"' Cargo.lock

::

    name = "rgmin"
    version = "0.2.0"
    source = "git+https://github.com/OmniPotentRPC/rgmin?branch=develop#06a41e5bbcef1a9c042cce7476cb6cba164db594"

In the rgmin checkout that ``Cargo.lock`` names, ``git rev-parse HEAD`` prints the same full commit.

Put the version in the entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give the entry a key, and add the version and its year.
The ``plain`` style of BibTeX prints a ``note`` field, but it drops ``version`` and ``url``, so put the version in ``note``.
Braces around ``rgmin`` in the title stop a style from capitalizing the name.
The 0.5.2 release reached crates.io on 25 October 2026:

.. code:: bibtex

    @misc{rgmin,
    author = {Goswami, Rohit},
    title = {{rgmin}: local minimization over eindir differentiable objectives},
    url = {https://github.com/OmniPotentRPC/rgmin},
    note = {Version 0.5.2},
    year = {2026}
    }

On the ``develop`` route, the note names the full commit, such as ``note = {Commit 06a41e5bbcef1a9c042cce7476cb6cba164db594 of the develop branch}``.

Cite the published work too
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods in rgmin come from published work.
The explanation pages name the book or paper that each derivation follows, and :doc:`../reference/bibliography` lists the works that the docs cite.
Cite the sources of the methods you ran, along with rgmin.
