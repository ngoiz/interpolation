# State-space interpolation for [SHARPy](http://www.github.com/ImperialCollegeLondon/sharpy)

A package for linear state-space interpolation and adaptive sampling of the training points based on Bayesian optimisation.

**Alpha Version - Development Only - No Support**

## Installation 

```bash
git clone http://github.com/ngoiz/interpolation
cd interpolation
pip install .
```

Required packages
- SHARPy
- GPyOpt
- PyDOE2

## User guide

The program runs from the command line
```
interpolation <path_to_settings_yaml_file>
```

or, alternatively from a Python script
```
interpolation_main.main(settings={}) # {} dictionary of settings
```


### State-space interpolation
Interpolates between state-spaces in an p-dimensional parameter space.

### Adaptive sampling based on Bayesian optimisation

Finds the optimal training points that minimise the interpolation error using Bayesian optimisation on different subsets of the available data in different combinations [1].


## References
[1] Goizueta, N., Wynn, A. and Palacios, R. Adaptive Sampling for Interpolation of Reduced-Order Aeroelastic Systems. AIAA Journal, Article in Advance, 2022. [doi](http://doi.org/10.2514/1.j062050)

