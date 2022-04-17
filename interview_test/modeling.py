from typing import Tuple, Dict
import re
import pickle

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import lightgbm as lgb


def train_lgb(x_train, y_train, x_test, y_test):
    """wrapper for lgb estimator"""
    eval_set = [(x_train, y_train), (x_test, y_test)]
    train_data = eval_set[0]

    lgb_estimator = lgb.LGBMRegressor(objective='multiclass',
                                      num_class=4,
                                      metric='multi_logloss',
                                      min_data_in_leaf=5,
                                      n_estimators=1000,
                                      max_depth=9,
                                      num_leaves=2 ** 9)

    lgb_estimator.fit(*train_data, eval_set=eval_set, early_stopping_rounds=50)
    return lgb_estimator


def stemming(text: str) -> str:
    """
    Remove all non letters chars and stop words from given string
    :param text: raw string
    :return: cleaned text
    """
    text_letters_only = re.sub('[^a-z]', ' ', text)
    splitted = text_letters_only.split()
    stemmed = [word for word in splitted if word not in stopwords.words('german')]
    return ' '.join(stemmed)


def save_model(model, path):
    """
    Save model to given folder
    :param model: ml model
    :param path: folder path where it should be saved
    :return:
    """
    pickle.dump(model, open(path, 'wb'))


def load_models(path: str) -> Tuple[TfidfVectorizer, lgb.LGBMClassifier, Dict[int, str]]:
    """
    Read saved models
    :param path: folder with saved model
    :return: tfidf model, lgb model and dict with clases
    """
    tfidf_model = pickle.load(open(f'{path}/tfidf_model.pickle', 'rb'))
    lgb_model = pickle.load(open(f'{path}/lgb_model.pickle', 'rb'))
    df = pd.read_csv(f'{path}/class_info.csv')
    clases_dict = {row['productgroup_id']: row['productgroup'] for _, row in df.iterrows()}
    return tfidf_model, lgb_model, clases_dict


def prepare_data_for_prediction(raw_str: str, tfidf_model: TfidfVectorizer) -> np.array:
    """
    Apply data preparing to single row and translate it to numeric vectors using tfidf_model
    :param raw_str: line with article text
    :param tfidf_model: tfidf model
    :return: array with features for further usage by lgb model
    """
    processed_str = raw_str.lower()
    processed_str = stemming(processed_str)
    features = tfidf_model.transform([processed_str]).toarray()
    return features


def generate_single_predictions(raw_str: str, path: str) -> str:
    """
    Aplly data preparation steps, read models, generate prediction for one single line
    :param raw_str: line with article text
    :param path: folder path where models are stored
    :return: predicted product group
    """
    tfidf_model, lgb_model, clases_dict = load_models(path)
    features = prepare_data_for_prediction(raw_str, tfidf_model)
    prediction = lgb_model.predict(features)
    higher_proba_pred = np.argmax(prediction)
    return clases_dict.get(higher_proba_pred)


def train_model_pipeline(data_path: str, model_path: str) -> None:
    """
    Pipeline from data reading to model saving
    :param data_path: path to the dataset
    :param model_path: folder where models will be stored
    :return:
    """
    # make sure that stopwords exists
    nltk.download('stopwords')

    df = pd.read_csv(data_path, sep=';', )

    # prepare column with numeric representation of the target
    df['productgroup_id'] = df['productgroup'].factorize()[0]

    # save mapping for further usage
    df[['productgroup_id', 'productgroup']].drop_duplicates().to_csv(f'{model_path}/class_info.csv', index=False)

    # data preparation part
    df.fillna('', inplace=True)
    # for now lets consider that all available parts can be used as single input string
    df['combined_text'] = pd.Series([' '.join(text) for text in df[['main_text', 'add_text', 'manufacturer']].values])
    df['combined_text'] = df['combined_text'].str.lower()
    df['combined_text'] = df['combined_text'].apply(stemming)

    # convert text features to numeric vectors using tfidf
    tfidf_model = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords.words('german'))
    features = tfidf_model.fit_transform(df['combined_text']).toarray()
    labels = df['productgroup_id']

    # split into train test and fit multi-class model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=0)
    lgb_model = train_lgb(X_train, y_train, X_test, y_test)

    # generate train test predictions
    pred_train = lgb_model.predict(X_train)
    pred_test = lgb_model.predict(X_test)

    # fetch value with max probability as prediction
    pred_one_train = [np.argmax(x) for x in pred_train]
    pred_one_test = [np.argmax(x) for x in pred_test]

    # display accuracy
    train_acc = accuracy_score(pred_one_train, y_train)
    test_acc = accuracy_score(pred_one_test, y_test)
    print(f'Training data accuracy: {train_acc}')
    print(f'Test data accuracy {test_acc}')

    # save models
    save_model(lgb_model, f'{model_path}/lgb_model.pickle')
    save_model(tfidf_model, f'{model_path}/tfidf_model.pickle')

    print(f'Models are saved to: {model_path}')
