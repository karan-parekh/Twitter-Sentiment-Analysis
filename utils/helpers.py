import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import seaborn as sns
import time

from textblob import TextBlob

from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from datetime import datetime
from emot import EMOTICONS
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVR
from typing import Optional
# from xgboost import XGBClassifier, XGBRegressor

from .paths import data_path


def get_subjectivity(text: str) -> str:
    sentiment = TextBlob(text).sentiment
    return sentiment.subjectivity

def get_polarity(text: str) -> str:
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity


def spell_check(text: str, blacklist=None) -> str:
    """Warning takes eternity to run"""

    if not blacklist:
        blacklist = []

    checker = SpellChecker()
    checker.word_frequency.load_words(blacklist)

    words     = text.split()
    corrected = [checker.correction(word) for word in words]

    return " ".join(corrected)


def save_as_csv(df: pd.DataFrame, filename: str=''):
    """Saves df to csv with unique filename"""
    time.sleep(1)  # to avoid over-writing files generated within a second
    
    if not filename:
        now = datetime.now()
        timestamp = datetime.strftime(now, "%d%m-%H%M%S")
        filename = "processed-{}.csv".format(timestamp)
    
    path = data_path(filename)
    
    print("Generating CSV at {}".format(path))
    
    df.to_csv(path, index=False, encoding='utf-8')
    print("Done")


def convert_emoticons(text: str) -> str:
    """Converts emoticons to words"""
    
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), text)

    return text


def get_hashtags(text, handle_space=True):

    if handle_space:
        space = r"#\s+"
        text  = re.sub(space, "#", str(text).lower()).strip()
    
    hash_tags = r"#\S+"
    alpha     = r"[^A-Za-z]"
    tags      = [re.sub(alpha, " ", tag).strip() for tag in re.findall(hash_tags, text)]

    return " ".join(tags)


def preprocess(tweet: str, stem=False, lema=False) -> str:
    """Remove links, @mentions, numbers and special characters from tweet"""

    TEXT_CLEANING_RE = r"@mention|https?:\S+|http?:\S|[^A-Za-z]+"
    stop_words       = stopwords.words("english")

    if stem:
        stemmer = SnowballStemmer("english")

    if lema:
        lematizer = WordNetLemmatizer()

    tweet  = re.sub(TEXT_CLEANING_RE, ' ', str(tweet).lower()).strip().replace("rt", "")
    tweet  = convert_emoticons(tweet)

    # print("Performing spell check")
    # tweet  = spell_check(tweet)

    tokens = []

    for token in tweet.split():

        if token not in stop_words:

            if lema:
                tokens.append(lematizer.lemmatize(token, pos='v'))

            elif stem:
                tokens.append(stemmer.stem(token))

            else:
                tokens.append(token)

    return " ".join(tokens)


def get_categorical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe with only categorical columns"""

    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    return df[cat_columns]


def get_numerical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe with only numerical columns"""

    num_columns = df.select_dtypes(include=np.number).columns
    return df[num_columns]


def missing_values_counts(df: pd.DataFrame) -> pd.Series:
    """Returns a Series of """

    counts = df.isnull().sum()
    return counts


def missing_values_percent(df: pd.DataFrame) -> pd.Series:
    """Returns a Series of """

    percent_null = (df.isnull().sum() / df.shape[0]) * 100
    missing_data = pd.Series(percent_null, index=df.columns)
    return missing_data


def show_skewness(df: pd.DataFrame):
    """Displays skew() for all continuous data in DataFrame"""

    df = get_numerical_df(df)

    for col in df.columns:
        print(col, df[col].skew())


def remove_right_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """Removes right skewness using np.log1p()"""

    df = get_numerical_df(df)

    for col in df.columns:

        if df[col].skew() > 1:
            df[col] = np.log1p(df[col])

    return df


def remove_left_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """Removes left skewness"""
    # To be defined


def plot_distributions(df: pd.DataFrame, color='#ffa600'):
    """Plots a distribution plot of all numerical columns using seaborn.distplots method"""

    cols = get_numerical_df(df).columns.tolist()

    for col in range(0, len(cols), 2):

        if len(cols) > col + 1:
            plt.figure(figsize=(10, 4))

            plt.subplot(121)
            sns.distplot(df[cols[col]], color=color)

            plt.subplot(122)
            sns.distplot(df[cols[col + 1]], color=color)

            plt.tight_layout()
            plt.show()

        else:
            sns.distplot(df[cols[col]], color=color)


def bar_plot_categorical_columns(df: pd.DataFrame):
    """PLots bar plot for all categorical columns"""

    cols = df.select_dtypes(include=['object']).columns

    for col in range(0, len(cols), 2):

        if len(cols) > col + 1:
            plt.figure(figsize=(10, 4))

            plt.subplot(121)
            df[cols[col]].value_counts(normalize=True).plot(kind='bar')
            plt.title(cols[col])

            plt.subplot(122)
            df[cols[col + 1]].value_counts(normalize=True).plot(kind='bar')
            plt.title(cols[col + 1])

            plt.tight_layout()
            plt.show()

        else:
            df[cols[col]].value_counts(normalize=True).plot(kind='bar')
            plt.title(cols[col])


def bivariate_analysis_categorical(df: pd.DataFrame, target_name: str):
    """Perform bivariate analysis for categorical columns"""

    target = df[target_name]
    df     = df.drop(target_name, 1)
    cols   = get_categorical_df(df).columns.tolist()

    for col in range(0, len(cols), 2):

        if len(cols) > col + 1:
            plt.figure(figsize=(15, 5))

            plt.subplot(121)
            sns.countplot(x=df[cols[col]], hue=target, data=df)
            plt.xticks(rotation=90)

            plt.subplot(122)
            sns.countplot(df[cols[col + 1]], hue=target, data=df)
            plt.xticks(rotation=90)

            plt.tight_layout()
            plt.show()


def bivariate_analysis_numerical(df: pd.DataFrame, target_name: str):
    """Perform bivariate analysis for numerical columns"""

    cols = get_numerical_df(df).columns.tolist()

    for col in cols:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df[target_name], y=df[col])
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()


def imbalance_percent(feature: pd.Series) -> int:
    """Returns percentage of imbalance in a categorical feature"""
    return (feature.value_counts() / feature.value_counts().sum()) * 100


def run_classification_models(X, y, models: Optional[dict], test_size=0.3, random_state=42) -> dict:
    """
    Runs a Baseline model on classification data
    Returns the AUC score
    """

    classification_models = {
        'Logistic Regression': LogisticRegression,
        'Decision Tree'      : DecisionTreeClassifier,
        'Random Forest'      : RandomForestClassifier,
        # 'XGBoost'            : XGBClassifier,
        'Gradient Boosting'  : GradientBoostingClassifier
    }

    if models:
        classification_models.update(models)

    def run_model(model) -> float:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        area_under_curve     = roc_auc_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

        print('F1_Score     : ', f1_score(y_test, y_pred))
        print('Recall Score : ', recall_score(y_test, y_pred))
        print('ROC_AUC_SCORE: ', area_under_curve)

        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.show()

        return area_under_curve

    auc_scores = {}

    for model_name, model_obj in classification_models.items():
        auc_scores[model_name] = run_model(model_obj)

    return auc_scores


def one_hot_encode(df: pd.DataFrame, columns=[]) -> pd.DataFrame:

    if not columns:
        columns = get_categorical_df(df).columns

    return pd.get_dummies(df, columns=columns)


def label_encode(df: pd.DataFrame, columns=[]) -> pd.DataFrame:

    if not columns:
        columns = get_categorical_df(df).columns
    
    encoder = LabelEncoder()

    for col in columns:
        df[col] = encoder.fit_transform(df[col])

    return df


def run_regression_models(X, y, models=None, scaler=None, test_size=0.3, random_state=42) -> dict:

    regression_models = {
        'Linear Regression' : LinearRegression,
        'Ridge'             : Ridge,
        'Lasso'             : Lasso,
        'Decision Tree'     : DecisionTreeRegressor,
        'Random Forest'     : RandomForestRegressor,
        'SVR'               : SVR,
        # 'XGBoost'           : XGBRegressor
    }

    if models:
        regression_models.update(models)
    
    def run_model(name, model):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        print(model_name + " : " + str(score))

        return model, score

    results = {}

    for model_name, model_obj in regression_models.items():
        model, score = run_model(model_name, model_obj())
        results[model_name] = {'model': model, 'score': score}

    return results


def run_model_on_test(model, test_df):

    y_pred = model.predict(test_df)
    return y_pred


def generate_heatmap(df: pd.DataFrame):
    corr = get_numerical_df(df).corr()
    sns.heatmap(corr, anot=True)


class RegressionException(Exception):
    """Raised when the regression methods encounters an Exception"""


if __name__ == "__main__":
    print("Done")
