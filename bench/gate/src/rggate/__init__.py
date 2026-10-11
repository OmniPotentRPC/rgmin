"""Statistical benchmark gate for repeated rgmin runs.

Outcome comparisons sit beside the Bayesian hierarchical performance
model of Goswami, AIP Adv. (2025), doi:10.1063/5.0283639.
Calls-to-success are right-censored and summarised by the Kaplan-Meier
estimator (JASA 53, 457, 1958, doi:10.1080/01621459.1958.10501452).
Solver curves are Dolan-More performance profiles (Math. Program. 91,
201, 2002, doi:10.1007/s101070100263), More-Wild data profiles (SIAM J.
Optim. 20, 172, 2009, doi:10.1137/080724083), and Gould-Scott pairwise
profiles when more than two methods are present (ACM TOMS 43, 2016,
doi:10.1145/2950048). Paired log call ratios use the BCa interval
(Efron, JASA 82, 171, 1987, doi:10.1080/01621459.1987.10478410).
The accept/reject rule is a normal-mixture confidence sequence
(Howard, Ramdas, McAuliffe, and Sekhon, Ann. Stat. 49, 1055, 2021,
doi:10.1214/20-AOS1991), the time-uniform form of a sequential test
(Wald, Ann. Math. Stat. 16, 117, 1945, doi:10.1214/aoms/1177731118).
"""

__version__ = "1"

from rggate.report import analyze, render

__all__ = ["analyze", "render", "__version__"]
