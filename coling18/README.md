# Emotion Representation Mapping: COLING 2018

## Introduction
See README of parent folder for a higher-level overview. 

## Installation

(If you are mainly interested in the emotion lexicons, you don't have to go through this.)

1. Costumize the paths in `EmoMap/coling18/framework/constants.py`
2. Set up the `conda` environment stored in `EmoMap/coling18/environment.yml`
3. Navigate into `EmoMap/coling18/` and run `source activate.src`. This will activate the `conda` environment and sets a required environment variable.
4. Congratulations! You're set-up to run any part of the codebase. Alternativly do `sh run_all.sh` to re-run everthing. Note that the results may be slighty off from the ones reported in the paper, since making the code completely deterministic is difficult due to multi-threading.

## Credits
Our code for word embeddings is loosely based on the [Hyperwords](https://bitbucket.org/omerlevy/hyperwords) package by [Levy et al. (2015)](https://aclanthology.coli.uni-saarland.de/papers/Q15-1016/q15-1016). The original code for the boosted multi-layer perceptron approach by [Du and Zhang (2016)](https://doi.org/10.1109/IALP.2016.7875958) can be found [here](https://github.com/StevenLOL/ialp2016_Shared_Task).

## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de
