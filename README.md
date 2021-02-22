# disaster_response

Disaster Response Project

This project is an introduction to Disaster Response Analysis.

First we're analyzing messages sent by population who suffered a disaster (hurricane or earthquake are examples)

## FILE DESCRIPTION

    .
    ├── app     
    │   ├── run.py                          # Flask file that runs app
    |   └── templates/
    │            ├── go.html                # Classification result page of web app
    │            └── master.html            # Main page of web app                                  
    ├── data                   
    │   ├── disaster_categories.csv         # Dataset including all the categories  
    │   ├── disaster_messages.csv           # Dataset including all the messages
    │   └── process_data.py                 # Data cleaning
    ├── models
    │   ├── train_classifier.py             # Train ML model
    │   └── classifier.pkl                  # pickle file of model   
    |   
    |── requirements.txt                    # contains versions of all libraries used.
    |
    └── README.md

## INSTALLATION

You need to download the .db which are the dataset to input in the model.

- process_data.py is the file where the messages are processed.
- train_classifier.py is the file to train the model (it can take several hours considering the grid search performed).

Dependencies:

    python (>= 3.6) https://www.python.org/downloads/
    pickle object serialization (available in python 3.6)
    pandas (>= 1.0.0) https://pandas.pydata.org/
    nltk (>= 3.5) https://www.nltk.org/
    scikit-learn (>= 0.23.2) https://scikit-learn.org

### Project motivation

For this project we're trying to predict the needs from population in the middle of disaster and connect people who offers help to people who needs help.

### Contribution

You're free to use and to contribute to the development of this project.

## INSTRUCTIONS

The following commands have to be executed in order to clean data and train a classifier

    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Then run the app to execute the front-end system classifier

    python run.py
    
Go to http://0.0.0.0:3001/


