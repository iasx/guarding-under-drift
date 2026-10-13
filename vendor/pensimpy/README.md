# pensimpy

The IndPenSim simulator as used in the article: PenSimPy (Quartic.ai, MIT licence, see `LICENSE`) ported to Python
3.12 and numpy 2, with a gymnasium wrapper; fixes to the recipe timing, disturbance seeding, offline assays and
episode reset; and options for a graded or removed dissolved-CO2 growth inhibition
(`pensimpy/ode/indpensim_ode_py.py`). Copied from https://github.com/iasx/PenSimPy at commit
`d0f2887d0945364777687569a67e5217a31ea2aa`, without its examples and tests.
