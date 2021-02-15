import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''Function to load data. messages_filepath and categories_filepath is
       ../directory/messages.csv and ../directory/categories.csv
    '''
    
    messages = pd.read_csv(str(messages_filepath), sep=',')
    categories = pd.read_csv(str(categories_filepath), sep=',')
    
    df = messages.merge(categories, how='inner', on='id')
    
    return df

def clean_data(df):
    
    '''Function to clean dataset, it gets the dataframe output from load_data() and returns the dataframe with the target
       splitted in one column each.
    '''
    
    categories = df.categories.str.split(';', expand=True)
    
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x[:-2])    
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    return df.drop_duplicates()


def save_data(df, database_filename):
    
    '''Function to save the datasets in a single SQL database.
    '''
    
    engine = create_engine('sqlite:///'+str(database_filename))
    df.to_sql('messages_etl', engine, index=False)

def main():
    
    ''' Main functio to call process_data process
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
