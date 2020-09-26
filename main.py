#! /usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def first_analysis():
    """
    Analyzes column correlations.

    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(train.corr(), vmin=-1, vmax=1, annot=True)
    plt.show()


def further_analysis():
    """
    Optional analysis. Used for visualizing other columns to optimize.

    """
    print(train[['SibSp', 'Parch']].info())
    train['SibSp'].value_counts().plot(kind='bar')
    plt.show()
    train['Parch'].value_counts().plot(kind='bar')
    plt.show()
    sib_pivot = train.pivot_table(index="SibSp", values="Survived")
    sib_pivot.plot.bar(ylim=(0, 1), yticks=np.arange(0, 1, .1))
    plt.show()
    parch_pivot = train.pivot_table(index="Parch", values="Survived")
    parch_pivot.plot.bar(ylim=(0, 1), yticks=np.arange(0, 1, .1))
    plt.show()
    explore_cols = ['SibSp', 'Parch', 'Survived']
    explore = train[explore_cols].copy()

    explore['family_size'] = explore[['SibSp', 'Parch']].sum(axis=1)
    # Create histogram
    explore['family_size'].value_counts(sort=False).plot(kind='bar')
    plt.show()
    family_pivot = explore.pivot_table(index=['family_size'], values="Survived")
    family_pivot.plot.bar(ylim=(0, 1), yticks=np.arange(0, 1, .1))
    plt.show()


def process_missing(df):
    """
    Converts missing data in the dataframe to values interpretable to ML
    models.

    :param df: Dataframe with missing values
    :return: Transformed dataframe
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


def process_age(df):
    """
    Converts the Age column in the dataframe to pre-defined bins.

    :param df: Dataframe
    :return: Dataframe with Age column having pre-defined bins
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ["Missing", "Infant", "Child", "Teenager", "Young Adult",
                   "Adult", "Senior"]
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


def process_fare(df):
    """
    Converts the Fare column into pre-defined bins.

    :param df: Dataframe
    :return: Dataframe with Fare column having pre-defined bins
    """
    cut_points = [-1, 12, 50, 100, 1000]
    label_names = ["0-12", "12-50", "50-100", "100+"]
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels=label_names)
    return df


def process_cabin(df):
    """
    Converts the Cabin column into pre-defined bins.

    :param df: Dataframe
    :return: Dataframe with Cabin column having pre-defined bins
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin', axis=1)
    return df


def process_titles(df):
    """
    Extracts and categorizes the title from each name entry.

    :param df: Dataframe
    :return: Dataframe with an additional column for a person's title
    """
    titles = {
        "Mr": "Mr",
        "Mme": "Mrs",
        "Ms": "Mrs",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Miss": "Miss",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Countess": "Royalty",
        "Dona": "Royalty",
        "Lady": "Royalty"
    }
    extracted_titles = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df


def process_family(df):
    """
    Evaluates the SibSp and Parch columns to determine if passenger was
    alone and assigns accordingly.

    :param df: Dataframe to be transformed
    :return: Transformed dataframe
    """
    df.loc[(df['SibSp'] > 0) | (df['Parch'] > 0), 'isalone'] = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'isalone'] = 1
    return df


def create_dummies(df, column_name):
    """
    Creates dummy columns (one hot encoding) from a single column

    :param df: Dataframe
    :param column_name: Column from the dataframe
    :return: Dataframe with the created dummy columns
    """
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([column_name], axis=1)
    return df


# Create a new function that combines all the functions in functions.py
def process_data(df):
    """
    Cleans and processes the dataframe to be ready for use in ML models.

    :param df: Original dataframe
    :return: Transformed dataframe
    """
    # Perform data cleaning
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)
    df = process_family(df)

    # Create binary classifications from columns & create dummy columns
    df = create_dummies(df, 'Age_categories')
    df = create_dummies(df, 'Fare_categories')
    df = create_dummies(df, 'Title')
    df = create_dummies(df, 'Cabin_type')
    df = create_dummies(df, 'Sex')

    return df


def select_features(df):
    """
    Selects features to use in Random Forest model.

    :param df: Clean dataframe
    :return: Columns predicted to provide best fit for data/predictions
    """
    numeric_df = df.select_dtypes(exclude=['object'])
    numeric_df = numeric_df.dropna(axis=1)
    all_x = numeric_df.drop(['PassengerId', 'Survived'], axis=1).copy()
    all_y = numeric_df['Survived'].copy()

    rfc = RandomForestClassifier(random_state=1)
    selector = RFECV(rfc, cv=10)
    selector.fit(all_x, all_y)
    best_columns = all_x.columns[selector.support_]

    return best_columns


def select_model(df, feature_list):
    """
    Provides a summary of ML models and hyperparameters that fit the
    training data.

    :param df: Clean dataframe
    :param feature_list: List of columns to use in model
    :return: Dictionary of tested ML models and hyperparameters
    """
    all_x = df[feature_list]
    all_y = df['Survived']

    models = [{
        'name': 'Logistic Regression',
        'estimator': LogisticRegression(),
        'hyperparameters':
            {
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [1000]
            }
    },
        {
            'name': 'K-Neighbors Classifier',
            'estimator': KNeighborsClassifier(),
            'hyperparameters':
                {
                    'n_neighbors': range(1, 20, 2),
                    'weights': ['distance', 'uniform'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]
                }
        },
        {
            'name': 'Random Forest Classifier',
            'estimator': RandomForestClassifier(),
            'hyperparameters':
                {
                    'n_estimators': [10, 25, 50, 100],
                    'criterion': ['entropy', 'gini'],
                    'max_depth': [2, 5, 10],
                    'max_features': ['log2', 'sqrt'],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 8],
                    'min_samples_split': [2, 3, 4, 5]
                }
        },
        {
            'name': 'Support Vector Classifier',
            'estimator': SVC(),
            'hyperparameters':
                {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }
        }
    ]

    for model in models:
        print(model['name'])
        grid = GridSearchCV(model['estimator'],
                            param_grid=model['hyperparameters'],
                            cv=5)
        grid.fit(all_x, all_y)

        model['estimator'] = grid.best_estimator_
        model['hyperparameters'] = grid.best_params_
        model['score'] = grid.best_score_

        print(model['hyperparameters'])
        print(model['score'])

    return models


def save_submission_file(model, columns, filename=None):
    """
    Uses a specified ML model to predict on holdout (test) data.
    Saves the results into a CSV file that can be submitted to Kaggle.

    :param model: ML model
    :param columns: List of columns to use in ML model
    :param filename: Specified filename. Default is to use the dataframe
    variable name.
    :return: CSV file containing passenger ID and predicted survival
    values on test data.
    """
    holdout_predictions = model.predict(holdout[columns])

    submission = {'PassengerId': holdout['PassengerId'],
                  'Survived': holdout_predictions
                  }
    submission = pd.DataFrame(submission)
    submission.to_csv(path_or_buf=filename, index=False)
    return print('Save successful!')


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    holdout = pd.read_csv('test.csv')

    first_analysis()
    further_analysis()

    # Pre-process data to prepare it for ML models
    train = process_data(train)
    holdout = process_data(holdout)

    selected_columns = select_features(train)
    print(selected_columns)

    # Automate grabbing the best model from the list of dictionaries
    selected_models = select_model(train, selected_columns)
    best_model = max(selected_models, key=lambda x: x['score'])
    print(best_model)

    best_model_estimator = best_model['estimator']

    # Create and save predictions to a CSV file
    save_submission_file(best_model_estimator,
                         selected_columns,
                         filename='predictions.csv')
