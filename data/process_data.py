# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """Function to load messages and categories datasets
    Parameters
    ----------
    messages_filepath : str
        Path to messages csv file
	categories_filepath: str
        Path to categories csv file
    Returns
    -------
    csv files as a data frame 
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')

    return df

def clean_data(df):
    """Function to clean the Pandas DataFrame created from the load_data function
    Parameters
    ----------
    df: pandas.DataFrame
        Messages and Categories Pandas DataFrame
    -------
    Pandas DataFrame
        Returns a clean Pandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # use the row to extract a list of new column names for categories
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0]).values
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis= 1)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    

def save_data(df, database_filename):
    """Function to save the clean dataframe to sqlite database
    Parameters
    ----------
    df: pandas.DataFrame
        clean Pandas DataFrame
	database_filename: str
        Path to sqlite database destination file
    Returns
    -------
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')

def main():
    """Main function to run the ETL pipeline
    Functions:
    1) Extraction -> extract messages and categories from csv files
	2) Transformation -> clean and pre-process data
	3) Load -> load the clean dataframe to SQLite local database
	"""
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()