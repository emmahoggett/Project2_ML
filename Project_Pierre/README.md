## Librairies

This version was designed for python 3.6.6, we used the following librairies:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [gensim](https://radimrehurek.com/gensim/): pip install gensim
* [scikit-learn](https://scikit-learn.org/stable/): pip install -U scikit-learn
* [joblib](https://pythonhosted.org/joblib/index.html): pip install joblib
* [scipy](https://www.scipy.org/): pip install scipy
* [keras](https://keras.io/): pip install Keras
* [tensor flow](https://www.tensorflow.org/install/): pip install tensorflow

[Fasttext](https://fasttext.cc/docs/en/supervised-tutorial.html) is also used. Tu use it, run the following:
* wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
* unzip v0.1.0.zip
* cd fastText-0.1.0
* make

## External datasets

We used word embeddings from external sources:
* [Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
* [Stanford](https://nlp.stanford.edu/projects/glove/)

## Where to put the data

For the Bigrams, GloVe, Word2Vect moving averages and FastText see the readme files in /Bigrams_MovingAverage, /GloVe_MovingAverage, /Word2Vec_MovingAverage and /FastText

The provided data as well as the external embeddings from Google and Stanford should be stored in /data.

## Generating a submission

To obtain the same submission as on crowdai:
* type `jupyter notebook` in the terminal
* open `run.ipynb`
* launch all cells in sequential order
* a file `LSTMSubmission.csv` will appear at the root, this is our submission

## File structure

run.ipynb: Used to generate our final submission

neural_networks.ipynb: Contains implementations for CNN, LSTM, GRU, RNN and RCNN classifiers as well as the stacking, SNN and majority voting. Also contains the code to generate those submissions.

FastText: Contains files for fasttext algorithm implementation (see readme contained inside the folder)

GloVe_MovingAverage: Contains files used to generate GloVe embeddings using a moving average. (see the readme inside)

Word2Vec_MovingAverage: Contains files used to generate Word2Vec embeddings using a moving average. (see the readme inside)

Bigrams_MovingAverage: Contains files used for the bigrams. (see the readme inside)
