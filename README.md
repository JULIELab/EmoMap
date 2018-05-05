Requirements: Code has been tested on MacOS with Python 3.5.1. The code is fairly lightweight and only requires common libraries such as Scipy and SciKit-Learn.

## Installation and Reproduction of Results
1. Set up an environment variable ```export BuechelLrec18Resources="/path/to/your/resources"```
2. Set up a conda environment ```conda create --name lrec18emotion --file environment.txt```
3. ```source activate_project_environment```
4. ```cd experiments```
5. ```python protocol.py``` 

Usage: Navigate to experiments folder and run 'python protocol.py' to re-run the experiments (takes about 30 sec on a laptop) or, in the same folder, run 'explore_english_lex.py' to run the exemplary analysis of the newly created lexicon.