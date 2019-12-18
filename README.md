# EPFL Machine Learning Recommender System 2019

### Description
This project's aim is to create predict good recommandation for films. Each user gives appreciations on films with grades that are integers between 0 and 5. One has to predict the remaining grades.

### Getting Started
This version was designed for python 3.6.6 or higher. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is `python run.py`. The code should return a `results.csv` file with all its predictions, from the test data.

### Prerequisites
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [surprise](https://surprise.readthedocs.io/en/stable/index.html): `pip install scikit-surprise`
* [scikit-learn](https://scikit-learn.org/stable/): `pip install -U scikit-learn`
* [keras](https://keras.io/): `pip install Keras`
* [tensor flow](https://www.tensorflow.org/install/): `pip install tensorflow`


To lunch the code `run.py` use the following codes and pickle files:
* `helpers.py` : Deal with creation and loading of `.csv` files
* `models/modelNN.py` : Contains methods for the neural network computations
* `models/modelSurprise.py`: Contains surprise methods
* `models/modelBaseline.py`: Contains baseline methods
* `models/modelMatrixFact.py`: Contains matrix factorization methods
* `data/test_pred.pickle`: Test file used for the Ridge cross validation model in `run.py`


The `data` folder is also needed to store training data and the data for the final submission : `data_train.csv` and `sampleSubmission.csv`.

### Additional content
The folder `scripts` contains python code that established our machine learning procedure,  contains the testing of the different methods implemented. To run those files, the same files as before are needed. scripts: 

### Documentation
* [Class Project 2](https://https://github.com/epfml/ML_course/tree/master/projects/project2/project_recommender_system) : Description of the project.
* [Resources](https://www.https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019/dataset_files): Datas for the training and testing.

### Authors
* Group name: [ML_Budget_3000](https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019/teams/ML_Budget_3000)
* Members: Aubet Louise, Cadillon Alexandre, Hoggett Emma

### Project Status
The project was submitted on the 19 December 2019, as part of the [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) course.
