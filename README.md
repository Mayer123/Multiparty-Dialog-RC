# Multiparty-Dialog-RC

This repository contains the code for the paper "Challenge Reading Comprehension on Daily Conversations: Passage Completion on Multiparty Dialog". See full paper [here](https://www.aclweb.org/anthology/N18-1185/)

## Requirements
* Python 2.7
* Numpy >= 1.13.3
* Tensorflow >= 1.4.0
* Keras >= 2.0.9

## Datasets

Our datasets with experimental splits can be found at [dialog_rc_data](dialog_rc_data) in json format.

The original TV show transcripts in json format can be found at [Character Mining](https://github.com/emorynlp/character-mining) project. 

Word embeddings: We used [Glove vectors](http://nlp.stanford.edu/data/glove.6B.zip) with 100 dimentions. 


## Usage
```
    python exp.py --train_file ../dialog_rc_data/json/Trn.json 
                   --dev_file ../dialog_rc_data/json/Dev.json 
                   --embedding_file glove.6B.100d.txt
                   --model cnn_lstm_UA_DA --logging_to_file log.txt
                   --save_model model.h5 --stopwords stopwords.txt
```


### Options
* `hidden_size`: default is 32.
* `batch_size`: default is 32.
* `utterance_filters`: default is 50.
* `query_filters`: default is 50.
* `nb_epoch`: default is 100.
* `dropout`: default is 0.2.
* `learning_rate`: default is 0.001.


## Cite 
```
@inproceedings{ma-etal-2018-challenging,
    title = "Challenging Reading Comprehension on Daily Conversation: Passage Completion on Multiparty Dialog",
    author = "Ma, Kaixin  and
      Jurczyk, Tomasz  and
      Choi, Jinho D.",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N18-1185",
    doi = "10.18653/v1/N18-1185",
    pages = "2039--2048",
}
```
