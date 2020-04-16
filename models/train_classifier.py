# import libraries

import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(database_filepath):
    """Function to load data from database
    Parameters
    ----------
    database_filepath : str
    Path to database file

    Returns
    -------
    Pandas DataFrames
        The dataframes consists of messages, categories features and category names
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)


    X = df['message']
    y = df.iloc[:, 4:]

    category_names = y.columns
    return X, y, category_names


def tokenize(text):

    """Function to tokenize text
    
    Parameters
    ----------
    text : str
        String to tokenize
    
    Returns
    -------
    List
        List of tokenized words
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
   
    for token in tokens:
        if token not in stopwords.words('english'):
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)
    
    return clean_tokens

def build_model():
    """Function to build a machine learning pipeline
    The pipeline takes in the message column as input and output classification results on the other 36 categories in the dataset
    Parameters
    ----------
    none
    Returns
    -------
    sklearn.pipeline.Pipeline
        sklearn.pipeline.Pipeline object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    # use the model to make predictions and convert it into a dataframe
    y_pred = model.predict(X_test)
    

    # calculate overall prediction accuracy
    overall_accuracy = (y_pred == y_test).mean().mean() * 100

    # convert y_pred to dataframe for convinience
    y_pred = pd.DataFrame(y_pred,columns = y_test.columns)

    data_array = []
    for col in y_test.columns:
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[col], y_pred[col], average='weighted')
        accuracy = np.mean(y_test[col].values == y_pred[col].values) * 100
        info = (col,precision, recall, f1_score,accuracy)
        data_array.append(info)
    
    # converting the result of evaluation metrics into a dataframe

    results = pd.DataFrame(data = data_array,columns=['Category', 'f1_score', 'precision', 'recall','accuracy'])
    
    print(results)
    
    # print overall model accuracy f1_score, precision and recall
    print('Aggregated f_score:', results['f1_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
     

    # print overall model accuracy
    print('Overall Accuracy: {0:.1f} %'.format(overall_accuracy))

def save_model(model, model_filepath):
    """Pickle model to the given file path
    Parameters
    ----------
    model   : model object
        fitted model
    model_filepath: str
        File path to save the model to
    Returns
    -------
    none
    """
    with open(model_filepath, 'wb') as f:
        # Pickle the 'model' to disk
        pickle.dump(model, f)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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

    
