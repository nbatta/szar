# szlib

## Changes

- Moved most of the classes to szlib/szcounts.py
- Moved the main iPython script to tests/testCounts.py
- Some functions referenced the global instantiation of its parent class instead of `self`. Changed those to `self`.
- Cosmology and constants read from an ini file and referenced from single class, less CAMB calls
- Numbafied some functions but not a huge difference
- Un-numbafiable bottleneck y_norm function sped up very slightly
- Added scripts for batch submission and processing on a cluster
- Added functionality to read SN from output of batch job


## Installation


To install while working on the repo (with symlinks):

```pip install -e .```

otherwise

```pip install .```

## A Dependency

I'm changing some of the plotting so that it uses a wrapper from this library
https://github.com/msyriac/orphics

It should install and work out of the box with the instructions there.

## Test

Use the command
```
python tests/testCounts.py
```

to run the same tests as in your original IPython notebook.
