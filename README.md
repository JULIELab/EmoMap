# Emotion Representation Mapping
This repository comprises code, experimental results and language resources associated with our [LREC 2018 paper](http://www.lrec-conf.org/proceedings/lrec2018/summaries/402.html) on converting between different emotion representation formats.

## Introduction
Emotion lexicons (data sets which describe the emotions which are associated with individual words) are an important resource in sentiment analysis. However, there are many different ways how affective states can be described, for example in terms of Basic Emotions are Valence-Arousal-Dominance. Having many of these so-called **emotion representation formats** brings up problems regarding comparability and inter-operability of different kinds of language resources (data sets as well as software tools). In order to address these problems, we propose a simple yet effective technique convert between different emotion formats so that, for example, an emotion lexicons which uses Basic Emotions can be translated into a Valence-Arousal-Dominance encoding, and the other way the round. We call this task **emotion representation mapping**. We evaluate our approach on a highly multilingual collection of data sets and find that it is about as reliable as human annotation. Based on these results we automatically create new emotion lexicons for a wide range of languages

## Folder Structure
This repository contains 
1. our full code base for an easy reproduction of our results,
2. additional experimental data, and most importantly
3. **high-quality emotion lexicons for 8 different languages**.

Our experimental code is located in the folder `experiments`. The folder `results` holds the experimental data of the three main experiments described in the paper. The folder `analysis` holds additional experimental as well data analyses results (inter-study reliabilities, tables of p-values, descriptive statistics of the created lexicons,...). The emotion lexicons are located in `resources`.

## Generated Emotion Lexicons
We distribute a total of nine emotion lexicons covering eight distinct languages (English, Spanish, Italian, Portuguese, German, Dutch, Polish, and Indonesian) as well as two emotion format, Valence-Arousal-Dominance and Basic Emotion (Joy, Anger, Sadness, Fear and Disgust). For both formats, our lexicons use **numerical scores**. The files can be found under the folder `resources` in tsv-format (using TAB as delimiter).
  
## Requirements, Installation and Reproduction of Results
Our code base has been developed and tested on MacOSX. However, it's fairly lightweigt and only requires common libraries such as Scipy and SciKit-Learn. Using it under linux should be no issue. Our dependencies are listed in `environment.txt`. To re-run our experiments, you need to gather the data sets we used, see `experiments/constants.py` (we do not distribute them directly due to legal restrictions):

1. Set up an environment variable `export BuechelLrec18Resources="/path/to/your/resources"` and place all the resources there,
2. create a conda environment form the descriptor file `conda create --name lrec18emotion --file environment.txt` (the file will need to updated for linux use),
3. activate the environment by running `source activate_project_environment`
4. `cd experiments`
5. run `python protocol.py`

The last command replicates all of our experiments and analyses. All random seeds are set so that there should be no changes in the output files. The script runs in about 30 seconds on a laptop.


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

## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de
