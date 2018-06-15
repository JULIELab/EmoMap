# Emotion Representation Mapping
This repository comprises code, experimental results and language resources associated with our [LREC 2018](http://www.lrec-conf.org/proceedings/lrec2018/summaries/402.html) and COLING 2018 papers on converting between different emotion representation formats.

## Introduction
Emotion lexicons (data sets which describe the emotions which are associated with individual words) are an important resource in sentiment analysis. However, there are many different ways how affective states can be described, for example in terms of Basic Emotions are Valence-Arousal-Dominance. Having many of these so-called **emotion representation formats** brings up problems regarding comparability and inter-operability of different kinds of language resources (data sets as well as software tools). In order to address these problems, we propose a simple yet effective technique convert between different emotion formats so that, for example, an emotion lexicons which uses Basic Emotions can be translated into a Valence-Arousal-Dominance encoding, and the other way the round. We call this task **emotion representation mapping**. We evaluate our approach on a highly multilingual collection of data sets and find that it is about as reliable as human annotation. Based on these results we automatically create new emotion lexicons for a wide range of languages

## Folder Structure
The two subfolders `lrec18`and `coling18`contain code, experimental results and our generated emotion lexicons for the respective paper. Both directories hold self-contained codebases which come with their own set-up instructions and environments.


## Citation
If you use our emotion lexicons or our code base, please cite our papers:

* Sven Buechel and Udo Hahn. 2018. Representation Mapping: A Novel Approach to Generate High-Quality Multi-Lingual Emotion Lexicons. In *LREC 2018 — Proceedings of the 11th International Conference on Language Resources and Evaluation*, pages 184 – 191, Miyazaki, Japan, May 7 – 12, 2018.

* Sven Buechel and Udo Hahn. 2018. Emotion Representation Mapping for Automatic Lexicon Construction (Mostly) Performs on Human Level. Accepted for *COLING 2018*.

Bibtex entries are provided in the subdirectories.

## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de
