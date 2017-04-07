# Tutorials for using the pipeline scripts

The pipeline generally consists of

0. Making an M,z grid of optical lensing deltaLnM once
1. Making an M,z grid of tSZ variances given an experiment and M,z grid spacings
2. Making an M,z grid of CMB halo lensing deltaLnM given an experiment and M,z grid spacings
3. Make "easy"-parameter derivatives given tSZ and lensing grids. Easy parameters can be varied by specifying a step size in `[params]`.
4. Make "hard"-parameter derivatives by running their respective scripts. e.g. sigma8(z), w_a
5. Calculate a Fisher matrix
6. Make 1D, 2D Fisher plots

## `input/pipeline.ini`

This ini file lets you specify configurations for experiments and Fisher matrices.

1. Identify or describe an experiment configuration (e.g. `[SO-6m]`)
2. Identify or describe a grid configuration (e.g. `[grid-default]`). This is the 
3. Make CMB lensing and SZ variance grids for each experiment by running `bin/makeGrid.py`. To see options for using it,
``
python bin/makeGrid.py -h
``

or do it in parallel with MPI using
``
python bin/subGrids.py
``
4. Make SZ grids for optical weak lensing
``
python bin/subOWLGrids.py
``
5. Make easy derivatives
6. Make sigma8 derivatives
7. Make CMB lensing offset derivatives
8. Make wa derivatives

