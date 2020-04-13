#import basics
import sys
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine


#import Scikit 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#download NLTK data
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_data(database_filepath):
    '''
    Loads data from a SQLite database, containing disaster messages
    
    Args:
    database_filepath: str -  the path for the sqlite databse
    
    Returns:
    X: str - dataframe containing text messages, independent variable
    Y: int - dataframe containing binary values for each category, dependent variables
    category_names: str - list of categories 
    
    '''
    name = 'sqlite:///'+database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('CleanDisasterMessages',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    '''
    Applies Natural Language Processing to raw text, namely: normalizes case, removes punctuation and english stop words, tokenizes and lemmatizes words.
    
    Args:
    text: str - raw message (text) to be cleaned
    
    Returns:
    tokens: cleaned, tokenized and lemmatized text
    '''
    
    #Normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]',' ' , text.lower())
    
    #Split text into words
    tokens = word_tokenize(text)
    
    # Initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    
    return tokens



def build_model():
    '''
    Builds a machine learning pipeline that takes in the message column as input and output classification results on the categories
    
    Args: non
    
    Returns:
    model: ML pipeline for text classification, optimized with GridSearchCv
    
    '''
    # ML Pipeline using Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters for GridSearchCv
    parameters = {#'clf__estimator__bootstrap': [True,False],
              #'clf__estimator__criterion': ['gini', 'entropy'],
              #'clf__estimator__n_estimators':[1,10,20,30,60],
              'clf__estimator__n_estimators': [10,30]
              # 'clf__estimator__n_estimators': [2]
             }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def calculate_stats(accuracy):
    '''
    Takes a list of accuracies and calculates the basic statistics, like minimum, maximum, mean and median
    
    Args:
    accuracy: str - list of accuracies for each category
    
    Returns: non
    '''
    minimum = accuracy.min()
    maximum = accuracy.max()
    mean = accuracy.mean()
    median = accuracy.median()
    
    print('Min:', minimum ,'\n','Max:', maximum ,'\n','Mean:', mean ,'\n','Median:', median)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performance by predicting on test data
    
    Arg:
    model: str - machine learning pipeline/model to be evaluated
    X_test: str - testing data for independent variables, messages
    Y_test: str - testing data for dependent variables, categories
    category_names: str - list of categories 
    
    Returns: non
    
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    
    #Full report: Accuracy, Precision, Recall and F1-score
    print ('Model Performance Metrics')
    
    for i in range(len(category_names)):
        print('Category:', category_names[i], '\n', classification_report(Y_test.iloc[:,1].values, y_pred[:,i])) 
    
    # Calculate accuracy for each category
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)

    #Apply calculate_stats function
    print(calculate_stats(accuracy))


def save_model(model, model_filepath):
    '''
    Saves trained model to a pickle file
    
    Args:
    model: str - ML pipeline/model to be saved
    model_filepath: str - file path to save the pickle file
    
    Returns: non
    
    '''
    #Save model to a pickle file
    filename = model_filepath
    pickle.dump(model, open(filename, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()