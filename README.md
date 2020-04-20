
# ItEM - Italian EMotive lexicon

ItEM is a a high-coverage emotion lexicon for Italian in which each target term is provided with an association score with the basic emotions defined the in Plutchik (1994)’s taxonomy: **JOY, SADNESS, ANGER, FEAR, TRUST, DISGUST, SURPRISE, ANTICIPATION**.

![alt text](images/1181px-Plutchik-wheel.png "Plutchik's wheel of emotions")

## Contents of this repository
 
 - the list of seed words collected in Passaro et al., 2015. The seeds are provided both as [lemmas](seeds/ItEM.elicitated.lemmas.txt) and [tokens](seeds/ItEM.elicitated.tokens.txt)
 
 - the [pre-compiled emotive lexicon](pre-compiled-lexica/ItEM.SintParModel.cos) described in Passaro and Lenci (2016) and referred as _SintParModel_ 
 
 
 - the [pre-compiled emotive lexicon](pre-compiled-lexica/ItEM.FBNEWS15.cos) built by exploiting count vectors extracted from FB-NEWS15 (Passaro et al., 2016); 
 
 - a [simplified implementation](ItEM.ipynb) of ItEM that can be used to create new lexica from a list of seeds and a list of word embeddings.

## Requirements and Usage

The code in this repository is compatible with Python3.x and depends on these libraries:

- Numpy
- Scipy
- Scikit-learn
- Gensim
- Pandas

We recommend to use a virtual environment and to install the specific versions of  each library provided in the [requirements file](requirements.txt)

Usage is described in the jupyter notebook [ItEM](ItEM.ipynb)