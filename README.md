# Udacity-DisasterResponse
Project 2 for Udacity Data Science NanoDegree

## Table of Contents

1. [Summary](#Summary)
2. [File Structure](#FileStructure)
3. [Required packages](#Requiredpackages)
4. [Instructions to run the App](#Instructions)
5. [Screenshots](#Screenshots)
6. [Acknowledgements](#Licensing)

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

1- **Distribution by Genre**  - shows the genre of the messages in the training set 
![Distribution by Genre](https://github.com/MargaridaFernandes1/Udacity-Project2-DisasterResponse/blob/master/DisasterResponse%20-%20Distribution%20by%20Genre.PNG?raw=true "Distribution by Genre")

2- **Number of Messages by Category** - shows the number of messages in each category

Note that the frequency of messages is quite different from one category to another. This means that, especially for the categories on the right side of the graph, the training set is imbalanced, threfore less instances to learn from. As a consequence, the accuracy for these categories is lower, as well as the quality of the final classification.

![Number of Messages by Category](https://github.com/MargaridaFernandes1/Udacity-Project2-DisasterResponse/blob/master/DisasterResponse%20-%20Count%20Messages.PNG)

3- **Top 10 Categories** - shows the top 10 categories with more messages
![Top 10 Categories](https://github.com/MargaridaFernandes1/Udacity-Project2-DisasterResponse/blob/master/DisasterResponse%20-%20Top%2010%20categories.PNG)

4- **Word Cloud** - represents the most used words in the disaster messages, where the size is proportional to its frequency
![Wordcloud](https://github.com/MargaridaFernandes1/Udacity-Project2-DisasterResponse/blob/master/DisasterResponse%20-%20WordCloud.PNG)

5- **Word Cloud** - Word cloud representation obtained with a different method in the ETL_Pipeline_Preparation.ipynb notebook
![Wordcloud notebook](https://github.com/MargaridaFernandes1/Udacity-Project2-DisasterResponse/blob/master/DisasterResponse%20-%20WordCloud%20(workspace).png)



## Acknowledgements<a name="Licensing"></a>
I would like to ackonwledge Udacity for providing materials as part of the Data Scientist Nanodegree, useful insights and interesting challenges like this project! And a thank you to Figure Eight for providing the datasets.
