
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

``rgmin`` is the local-minimization crate of the OmniPotentRPC family,
beside `rgpot <https://github.com/OmniPotentRPC/rgpot>`_ and `rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_. Algorithms live only in Rust. A session
takes one stepper step on an embedded manifold (``Euclidean``,
``MwRigid``, ``Sphere``, and the rest) and reports. Hosts own the loop.

C and C++ keep the old ``xts::optimize`` names through an hourglass C ABI
(``rgmin_solver_t`` / ``rgmin_minimize``) that carries **dlpk**
``DLManagedTensorVersioned`` tensors.
``include/xts/optimize.hpp`` is the C++ wrapper; ``include/xts/xtensor.hpp``
adapts ``xt::xarray``. There is no second solver implementation in C++.

The production unconstrained local method is limited-memory BFGS with
the strong Wolfe conditions (Nocedal and Wright 7.4 and algorithms
3.5--3.6). Hopping chains keep that solver's pair history on ``Lbfgs``;
anneal's ``WarmLbfgs`` is a thin handle around it.

Saddle, band, and IRC sessions live in ``rgsaddle``. They consume this
crate's ``Solver``; they do not grow a second optimizer.

.. code:: bash

    cargo add rgmin ndarray@0.17

That installs the 0.5.2 release.
:doc:`howto/install` covers the ``develop`` branch and how it differs from the release.
It also covers the features of each route, the C library and a checkout.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart
   tutorials/relax-morse

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   howto/install
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
