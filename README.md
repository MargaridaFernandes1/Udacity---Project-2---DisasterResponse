# Udacity-DisasterResponse
Project 2 for Udacity Data Science NanoDegree

## Table of Contents

1. [Summary](#Summary)
2. [File Structure](#File Structure)
3. [Required packages](#Required packages)
3.1 [Instructions to run the App](#Instructions to run the App)
4. [Screenshots](#Screenshots)
5. [Future Improvements](#Future Improvements)
6. [Licensing, Authors, and Acknowledgements](#Licensing, Authors, and Acknowledgements)

## Summary

When a disaster happens a huge amount of messages are exchanged on digital channels, such as social media. These messages contain extremely valuable information and can be used to mobilize the help of the competent entities. The objective of this project is to help in the classification of these messages through Natural Language Processing and Machine Learning techniques, in order to effectively deliver them to the responsible organizations.



## File Structure

File Structure of the application
repository data structure tree

## Required packages
write something

### Instructions to run the App

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots
The type of visualizations available on the web app

## Future Improvements
Write something

## Licensing, Authors, and Acknowledgements
write something
