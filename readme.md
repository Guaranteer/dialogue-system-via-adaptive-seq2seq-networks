A dialog system via adaptive seq2seq network
==========

## how to run

Download a word2vec file by yourself and put it in word2vec folder.

I use the GOOGLE NEWS  word2vec file which can be downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Then you can run the code from trainers folder.

For example, python trainer_HRED.py

Of source, you need tensorflow >= 1.3 and some necessary python packet.


## main algorithm

I'm lazy to describe it now.


## baseline

* seq2seq
* seq2seq with attention
* HRED
* VHRED (not standard)



## directory

* configs：configure files
* models: model files
* trains: train files
* utils: util files
* dataloaders: data loader files (change according to different dataset)



## dataset 

Please obtain dataset yourself and code a dataloader for it.

I use cornell movie corpus and reconstruct it.



## other

The code is about my research and I only save it on github. So it is very rough and maybe difficult to understand. 

xixi.


