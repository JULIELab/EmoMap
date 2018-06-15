# Emotion Representation Mapping
This repository comprises code, experimental results and language resources associated with our [LREC 2018](http://www.lrec-conf.org/proceedings/lrec2018/summaries/402.html) and [COLING 2018](https://www.researchgate.net/publication/325794428_Emotion_Representation_Mapping_for_Automatic_Lexicon_Construction_Mostly_Performs_on_Human_Level) papers on converting between different emotion representation formats.

## Introduction
Emotion lexicons (data sets which describe the emotions which are associated with individual words) are an important resource in sentiment analysis. However, there are many different ways how affective states can be described, for example in terms of Basic Emotions are Valence-Arousal-Dominance. Having many of these so-called **emotion representation formats** brings up problems regarding comparability and inter-operability of different kinds of language resources (data sets as well as software tools). In order to address these problems, we propose a simple yet effective technique convert between different emotion formats so that, for example, an emotion lexicons which uses Basic Emotions can be translated into a Valence-Arousal-Dominance encoding, and the other way the round. We call this task **emotion representation mapping**. We evaluate our approach on a highly multilingual collection of data sets and find that it is about as reliable as human annotation. Based on these results we automatically create new emotion lexicons for a wide range of languages

## Emotion Lexicons
The latest version of our automatically generated emotion lexicons cover a total of 13 languages. Most of them describe the words in terms  of five basic emotions categories (joy, anger, sadness, fear and disgust) on a numerical 5-point scale. They complement existing emotion lexicons which only describes the respective words according to another *emotion representation format* (Valence-Arousal or Valence-Arousal-Domaninance). The size of the generated emotion lexicons ranges up to 13k words. Details of our acquisition methodology is given in our COLING 2018 paper. Our results show, that these data sets, although automatically constructed, are virtually as reliable as manually annotated data. The indivual lexicons are listed below:

* [English](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/Warriner_BE.tsv)
* [Spanish](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/Stadthagen_Dominance.tsv)
* [German](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/Vo_BE.tsv)
* [Polish](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/Imbir_BE.tsv)
* [Italian](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/it_Montefinese_BE.tsv)
* [Portuguese](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/pt_Soares_BE.tsv)
* [Dutch](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/nl_Moors_BE.tsv)
* [Indonesian](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/id_Sianipar_BE.tsv)
* [Chinese](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/zh_Yu_Yao_BE.tsv)
* [French](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/fr_Monnier_BE.tsv)
* [Greek](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/gr_Palogiannidi_BE.tsv)
* [Finnish](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/fn_Eilola_BE.tsv)
* [Swedish](https://github.com/JULIELab/EmoMap/blob/master/coling18/main/lexicon_creation/lexicons/sv_Davidson_BE.tsv)

## Citation
If you use our emotion lexicons or our code base, please cite our papers:

* Sven Buechel and Udo Hahn. 2018. Representation Mapping: A Novel Approach to Generate High-Quality Multi-Lingual Emotion Lexicons. In *LREC 2018 — Proceedings of the 11th International Conference on Language Resources and Evaluation*, pages 184 – 191, Miyazaki, Japan, May 7 – 12, 2018.

* Sven Buechel and Udo Hahn. 2018. Emotion Representation Mapping for Automatic Lexicon Construction (Mostly) Performs on Human Level. Accepted for *COLING 2018*.

Bibtex entries are provided in the subdirectories.

## Folder Structure
The two subfolders `lrec18`and `coling18`contain code, experimental results and our generated emotion lexicons for the respective paper. Both directories hold self-contained codebases which come with their own set-up instructions and environments.


## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de
