# Emotion Representation Mapping
This repository comprises code, experimental results and language resources associated with our [LREC 2018 paper](www.lrec-conf.org/proceedings/lrec2018/summaries/402.html) on converting between different emotion representation formats.

## Citation
If you use our emotion lexicons or our code base, please cite our paper:

*Sven Buechel & Udo Hahn. 2018. Representation Mapping: A Novel Approach to Generate High-Quality Multi-Lingual Emotion Lexicons. In LREC 2018 — Proceedings of the 11th International Conference on Language Resources and Evaluation. Miyazaki, Japan, May 7 – 12, 2018. Pages 184 – 191.*

```
@InProceedings{BUECHEL18.402,
  author = {Sven Buechel and Udo Hahn},
  title = {Representation Mapping: A Novel Approach to Generate High-Quality Multi-Lingual Emotion Lexicons},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {may},
  date = {7-12},
  location = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {979-10-95546-00-9},
  language = {english}
  }
  ```

Requirements: Code has been tested on MacOS with Python 3.5.1. The code is fairly lightweight and only requires common libraries such as Scipy and SciKit-Learn.

## Requirements, Installation and Reproduction of Results
1. Set up an environment variable `export BuechelLrec18Resources="/path/to/your/resources"`
2. Set up a conda environment `conda create --name lrec18emotion --file environment.txt`
3. `source activate_project_environment`
4. `cd experiments`
5. `python protocol.py`

Usage: Navigate to experiments folder and run 'python protocol.py' to re-run the experiments (takes about 30 sec on a laptop) or, in the same folder, run 'explore_english_lex.py' to run the exemplary analysis of the newly created lexicon.

## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de
