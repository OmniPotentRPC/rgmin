
===============
rgmin
===============

:Author: `Rohit Goswami <https://rgoswami.me>`_

.. raw:: html

   <p class="rgmin-hero">
     <img class="rgmin-hero-logo rgmin-hero-logo--light"
          src="_static/rgmin-logo-light.webp"
          alt="rgmin logo"
          width="420"
          height="84"
          loading="eager" />
     <img class="rgmin-hero-logo rgmin-hero-logo--dark"
          src="_static/rgmin-logo-dark.webp"
          alt="rgmin logo"
          width="420"
          height="84"
          loading="eager" />
   </p>

Overview
--------

``rgmin`` finds a local minimum of a smooth function from its value and gradient.
A host program owns the loop, and a ``Solver`` session takes one step each time the host calls it.
The step can stay on an embedded manifold (``Euclidean``, ``MwRigid``, ``Sphere``, and the rest).
It is the local-minimization crate of the OmniPotentRPC family, beside `rgpot <https://github.com/OmniPotentRPC/rgpot>`_ and `rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_.

The solvers are written only in Rust.
C and C++ call them through one small C application binary interface (ABI), ``include/rgmin.h`` (``rgmin_solver_t`` / ``rgmin_minimize``).
Vectors cross it as DLPack ``DLManagedTensorVersioned`` tensors from the **dlpk** crate.
``include/rgmin/optimize.hpp`` is the C++ wrapper and ``include/xts/xtensor.hpp`` adapts ``xt::xarray``.
``include/xts.h`` and ``include/xts/optimize.hpp`` keep the old ``xts`` names for existing hosts.
There is no second solver implementation in C++.
:doc:`reference/architecture` draws these layers.

The production unconstrained local method is limited-memory BFGS with
the strong Wolfe conditions (Nocedal and Wright 7.4 and algorithms
3.5--3.6). Hopping chains keep that solver's pair history on ``Lbfgs``;
anneal's ``WarmLbfgs`` is a thin handle around it.

Saddle, band, and IRC sessions live in ``rgsaddle``. They consume this
crate's ``Solver``; they do not grow a second optimizer.

.. code:: bash

    git clone https://github.com/OmniPotentRPC/rgmin.git
    cd rgmin
    # Build and test on the remote builder, not a laptop.
    cargo test --features capi

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart
   tutorials/relax-morse

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   howto/session
   howto/manifolds
   howto/irc
   howto/embed-anneal
   howto/highs-qp
   howto/c-abi
   howto/doxygen

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/linesearch
   explanation/quasi-newton
   explanation/trust-region
   explanation/scg
   explanation/randomized

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/architecture
   reference/api
   reference/sota
   reference/proofs
   reference/bibliography
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing/index

License
-------

MIT.
