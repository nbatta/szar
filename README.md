# szlib

## Changes

- Moved most of the classes to szlib/szcounts.py
- Moved the main iPython script to tests/testCounts.py
- Some functions referenced the global instantiation of its parent class instead of `self`. Changed those to `self`.

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
