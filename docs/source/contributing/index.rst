

Contributing
------------

Report a bug or ask a question on the `issue tracker <https://github.com/OmniPotentRPC/rgmin/issues>`_.
Report a vulnerability as the `security policy <https://github.com/OmniPotentRPC/rgmin/blob/develop/SECURITY.md>`_ says, not in a public issue.
Everyone who takes part follows the `code of conduct <https://github.com/OmniPotentRPC/rgmin/blob/develop/CODE_OF_CONDUCT.md>`_, in issues and pull requests alike.

Pull requests go to the ``develop`` branch.
Commit subjects take the form ``type(scope): summary``, as in ``fix(session): report a stall when the energy test refuses a restart``.
A change that users can see adds a fragment to ``docs/newsfragments/`` and nowhere else.
The Unreleased sections of ``CHANGELOG.md`` and ``docs/orgmode/changelog.org`` stay empty, so ``towncrier build`` does not list an entry twice.
The release commit copies the notes into the org page under a heading for the version.
``python3 scripts/check_changelog.py`` fails when a fragment is paired with an Unreleased line.
Name the fragment ``<issue>.<type>.md``, or ``+<name>.<type>.md`` for a change with no issue.
The type is ``added``, ``changed``, ``deprecated``, ``removed``, ``fixed`` or ``security``, the sections of ``CHANGELOG.md``.
Write it as one sentence per line, the way ``CHANGELOG.md`` reads.
``towncrier build`` moves the fragments into ``CHANGELOG.md`` when a release is cut, with the settings in ``towncrier.toml``.
To see the notes that the fragments give without changing a file:

..
   #skip_lint_start

.. code:: bash

    python3 -m pip install towncrier
    towncrier build --draft --version 0.0.0

..
   #skip_lint_end

Build
~~~~~

Continuous integration (CI) runs these checks with Rust 1.88.0 on every push and pull request to ``main`` and ``develop``.
Run them before you open a pull request.

..
   #skip_lint_start

.. code:: bash

    rustup toolchain install 1.88.0 --profile minimal -c rustfmt
    python3 -m pip install numpy scipy
    cargo +1.88.0 fmt --all -- --check
    cargo +1.88.0 test --locked
    cargo +1.88.0 test --locked --features capi
    cargo +1.88.0 test --locked --features highs
    cargo +1.88.0 test --locked --features capi,highs
    cargo +1.88.0 test --locked --features capi,par,highs,slepc,primme,chase,libkrylov
    cargo +1.88.0 test --locked --features readcon
    cargo +1.88.0 test --locked --features periodic-product

..
   #skip_lint_end

CI calls the fifth row all-features.
It names the features instead of passing ``--all-features``, so the static library does not link ``readcon-core`` or the LGPL ``gmp-mpfr-sys``, which get the last two rows.
The ``periodic-product`` feature cannot be combined with ``capi``, and its row needs ``m4``, ``make``, ``diffutils`` and a C compiler.

The rows with ``capi`` also compile C and C++ programs against ``include/`` and run them.
Those tests run ``cargo build --lib`` for the target, profile and features of the test run and link the library it reports, so no ``cargo build`` has to come first.
They compile with ``CC_<target>`` and ``CXX_<target>``, or ``CC`` and ``CXX``, read the way the ``cc`` crate reads them.
Without those, they use the first of ``cc``, ``gcc`` and ``clang``, and of ``c++``, ``g++`` and ``clang++``, that builds a test program.
On Windows with MSVC they call ``cl``, so run them from a Visual Studio developer prompt.
Under ``--target`` they default to the triple without its vendor and ``-gcc`` or ``-g++``, such as ``aarch64-linux-gnu-gcc``, and run the programs through ``CARGO_TARGET_<TRIPLE>_RUNNER`` when it is set.
With or without ``--target``, a command in ``RGMIN_C_TEST_RUNNER``, such as ``valgrind --error-exitcode=99``, runs the programs in place of that runner, as in the capi row of CI.

CI also runs clippy with warnings denied, on Rust 1.88.0 and on the latest stable, for the first five feature sets:

..
   #skip_lint_start

.. code:: bash

    rustup component add clippy --toolchain 1.88.0
    cargo +1.88.0 clippy --locked --all-targets -- -D warnings
    cargo +1.88.0 clippy --locked --all-targets --features capi -- -D warnings
    cargo +1.88.0 clippy --locked --all-targets --features highs -- -D warnings
    cargo +1.88.0 clippy --locked --all-targets --features capi,highs -- -D warnings
    cargo +1.88.0 clippy --locked --all-targets --features capi,par,highs,slepc,primme,chase,libkrylov -- -D warnings

..
   #skip_lint_end

Run the same five commands with ``+stable`` for the second toolchain.

CI also builds and tests in release with Rust 1.93.0 and 1.94.1, which ship LLVM 21.
Its loop vectorizer once crashed on rgmin, and the debug builds above never run it:

.. code:: bash

    rustup toolchain install 1.93.0 --profile minimal
    cargo +1.93.0 build --locked --release
    cargo +1.93.0 test --locked --release --lib

Install 1.94.1 the same way and run the two ``cargo`` commands with ``+1.94.1``.
The default feature set is enough: ``--all-features`` on this line also links SLEPc, PRIMME, ChASE and libkrylov.

These checks need network access on their first run, since cargo fetches two git dependencies pinned to a commit: ``eindir-core`` and, for the tests, ``rgpot-core``.
The ``highs`` tests check against SciPy through ``python3``, or through the interpreter that ``PYTHON`` names.
The ``highs`` feature builds HiGHS from source, which needs CMake and a C++ compiler.

CI also installs the C library with cargo-c and compiles a C program and a C++ program against it.
cargo-c 0.10.17 needs Rust 1.89 or newer, so CI installs it with Rust 1.89.0 and then builds rgmin with 1.88.0.
cargo-c links against OpenSSL, the Transport Layer Security (TLS) and Secure Sockets Layer (SSL) library.
CI therefore installs ``libssl-dev`` and ``pkg-config`` first.
Without ``libssl-dev``, ``cargo install cargo-c`` exits 101, and the smoke script calls ``pkg-config`` itself.
On Ubuntu, as on the CI runner:

..
   #skip_lint_start

.. code:: bash

    sudo apt-get update
    sudo apt-get install -y libssl-dev pkg-config
    rustup toolchain install 1.89.0 --profile minimal
    cargo +1.89.0 install cargo-c --locked --version 0.10.17+cargo-0.92.0
    RUSTUP_TOOLCHAIN=1.88.0 tests/cinstall/smoke.sh

..
   #skip_lint_end

CI builds the LAMMPS plugin in ``lammps/`` against the 22Jul2025 and 30Sep2026 stable releases, each built from source with MPI and the ``MANYBODY`` and ``PLUGIN`` packages.
It runs ``lammps/tests`` with pytest through ctest, on 1, 2 and 4 MPI ranks.
With such a LAMMPS tree in ``$LAMMPS``:

..
   #skip_lint_start

.. code:: bash

    cargo +1.88.0 build --release --locked --lib --features capi
    cmake -S lammps -B build-lammps -D LAMMPS_SOURCE_DIR=$LAMMPS/src \
      -D RGMIN_LIBRARY=$PWD/target/release/librgmin.a -D RGMIN_LAMMPS_LMP=$LAMMPS/build/lmp
    cmake --build build-lammps
    LAMMPS_POTENTIALS=$LAMMPS/potentials ctest --test-dir build-lammps --output-on-failure

..
   #skip_lint_end

Citations
~~~~~~~~~

Narrative pages cite through ``sphinxcontrib-bibtex`` (``:cite:`key```) against ``docs/source/references.bib``.
That file is the ``rgmin-docs`` ookcite collection (``export_collection``).
Add a digital object identifier (DOI) with ``validate_doi`` then ``add_to_collection``; never edit the .bib, never invent keys, never keep a markdown citation table.
The rendered list is :doc:`reference/bibliography <../reference/bibliography>`.

Formulae
~~~~~~~~

Nocedal-Wright equation numbers are the contract:

- NLCG is algorithm 5.4

- Inverse BFGS is 6.17

- L-BFGS two-loop is 7.4, scale is 7.20

- Strong Wolfe is 3.5 / 3.6

- Goldstein is 3.11

- Hager-Zhang beta is paper equation 1.3, not the C++ port

Docs
~~~~

Narrative pages live in ``docs/orgmode/``, sorted by Diataxis into tutorials, how-to guides, explanation and reference.
Edit the ``.org`` page.
Sphinx reads its reStructuredText export from ``docs/source/``.
The docs workflow runs these steps, and you can run them from the top of the checkout:

..
   #skip_lint_start

.. code:: bash

    python3 -m pip install "sphinx>=8,<8.1" shibuya sphinxcontrib-bibtex myst-parser
    (cd docs && emacs --batch -l export.el)
    python3 docs/render_performance.py
    sphinx-build -b html -n docs/source docs/build

..
   #skip_lint_end

Run the export inside ``docs/``.
``export.el`` reads ``./orgmode/`` and writes ``./source/``, so run from anywhere else it writes nothing and still exits 0.
The export needs Org 9.7 or newer: ox-rst calls ``org-element-type-p``, which that release added, and the Emacs on the docs runner comes with Org 9.6.
``export.el`` installs a missing Org from the Emacs Lisp Package Archive (ELPA) of the GNU's Not Unix (GNU) project, and a missing ox-rst from Milkypostman's ELPA.
Its first run therefore needs network access.
It rewrites every page under ``docs/source/``, including pages whose committed copy has drifted from the Org source.
Commit the export of the pages you edit, and leave the others out.

Link to another page with ``@@rst::doc:`text <path>`@@``.
A ``file:`` link would point at the ``.rst`` file, which the built site does not have.

The site uses the Shibuya theme; logos live in ``branding/logo/`` and are copied into ``docs/source/_static/``.
The pages for the C application binary interface (ABI) are built with Doxygen plus doxyYoda; see :doc:`howto/doxygen <../howto/doxygen>`.

No workflow runs Doxygen, so build those pages by hand with the steps in that how-to.
Pull requests that touch ``docs/`` or the docs workflow build the narrative pages: they run the Org export and the Sphinx build.
The Pages deploy runs on a push to ``main`` whose paths touch ``docs/`` or the workflow, and on ``workflow_dispatch``.
A pull request has its own concurrency group, so it does not cancel a deploy.
