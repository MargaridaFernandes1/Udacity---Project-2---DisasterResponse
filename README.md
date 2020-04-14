# Udacity-DisasterResponse
Project 2 for Udacity Data Science NanoDegree

## Table of Contents

1. [Summary](#Summary)
2. [File Structure](#FileStructure)
3. [Required packages](#Requiredpackages)
4. [Instructions to run the App](#Instructions)
5. [Screenshots](#Screenshots)
6. [Future Improvements](#FutureImprovements)
7. [Licensing, Authors, and Acknowledgements](#Licensing)

## Summary<a name="Summary"></a>

When a disaster happens a huge amount of messages are exchanged on digital channels, such as social media. These messages contain extremely valuable information and can be used to mobilize the help of the competent entities.

The objective of this project is to help in the classification of these messages through Natural Language Processing and Machine Learning techniques, in order to effectively deliver them to the responsible organizations. This project includes:

-	An ETL pipeline based on the dataset provided by Figure Eight, containing the labelled data;
-	A ML pipeline, where NLP features where created and used to train a multi-output Random Forest Classifier. The classifier was then improved with Grid Search.
-	A Web App, using Flask library. The app allows you to insert a new message and returns the corresponding categories in real time. It also reads the original dataset and provides some descriptive visualizations. 



## File Structure <a name="FileStructure"></a>

The repository files have the following structure:
```text
Udacity_Project2-DisasterResponse/
└── app/
    └── templates/
        ├── go.html
        ├── main.html
    ├── run.py
└── data/
    ├── DisasterResponse.db
    ├── disaster_categories.csv
    ├── disaster_messages.csv
    ├── process_data.py
└── models/
    ├── train_classifier.py
    ├── classifier.pkl
└── ETL Pipeline Preparation.ipynb
└── ML Pipeline Preparation.ipynb
└── README.md
    
```
I. app
    * _templates:_ html files for the web application
    * _run.py:_ file to run the web app
II. data
    * _DisasterResponse.db:_ SQLite database with cleaned data; results as output of _process_data.py_
    * _disaster_categories.csv:_ dataset including all categories
    * _disaster_messages.csv:_ dataset including all messages
    * _process_data.py:_ python script containing the ETL pipeline: read, clean and save data into a database
III. models
    * _train_classifier.py:_ python script containing the ML pipeline: loads data, applies NLP techniques and builds a model. The output is a classifier in a .pkl file. 
    * _classifier.pkl:_ final classifier to classify new messafes, as output of _train_classifier.py_
IV. _ETL Pipeline Preparation.ipynb:_ Jupyter notebook for the ETL pipeline preparation
V. _ML Pipeline Preparation.ipynb:_ Jupyter notebook for the ML pipeline preparation and performance comparison of different classifiers

## Required packages<a name="Requiredpackages"></a>

The following packages must be installed:

Python
- Libraries: pandas, numpy, sklearn, sqlite3, sqlalchemy, nltk, plotly, flask, regex
HTML
- Bootstrap

### Instructions to run the App<a name="Instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots<a name="Screenshots"></a>

When the app is launched successfully, the following visualizations are available:

## Licensing, Authors, and Acknowledgements<a name="Licensing"></a>
I would like to ackonwledge Udacity for providing materials as part of the Data Scientist Nanodegree, useful insights and interesting challenges like this project! And a thank you to Figure Eight for providing the datasets.
