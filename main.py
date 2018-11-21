from feature_selector import FeatureSelector
import pandas as pd
import numpy as np
from svm import SVMModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Chọn chức năng')
parser.add_argument('func', nargs='?',
                    help='func name',
                    default='validate')
args = parser.parse_args()

def load_data():
    data = pd.read_csv('data/user_example_features_selected.csv').dropna()
    train_labels = data['TARGET'].values
    train = data.drop(['UserID', 'TARGET'], axis=1)
    feat_names = data.columns
    return train.values, train_labels, feat_names

def select_features():
    # df = df.fillna(0)
    data = pd.read_csv('data/user_example.csv').dropna()
    # for column in data:
    #     print(data[column].dtypes)

    # data = pd.to_numeric(data)
    train_labels = data['TARGET']
    user_ids = data['UserID']
    # labels = data.as_matrix(columns=[train_labels])
    train = data.drop(['UserID', 'TARGET'], axis=1)

    fs = FeatureSelector(data = train, labels = train_labels)
    fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                        'task': 'classification', 'eval_metric': 'auc', 
                                        'cumulative_importance': 0.99})
    train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)
    # for column in train_removed_all_once:
    #     print(train_removed_all_once[column].dtypes)
    # print(fs.feature_importances.head())
    train_removed_all_once.loc[:,'UserID'] = user_ids.values
    train_removed_all_once.loc[:,'TARGET'] = train_labels.values
    train_removed_all_once.to_csv('data/user_example_features_selected.csv')

def train(X, y):
    model = SVMModel()
    model.fit(X, y)

def validate(X, y, feat_names, feat1_index = 3, feat2_index = 9):
    model = SVMModel()
    model.load()
    pred = model.predict(X)
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.5)
    plt.title("t-sne")
    # plt.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)
    plt.show()

    feat1 = feat_names[feat1_index]
    feat2 = feat_names[feat2_index]
    plt.scatter(X[:, feat1_index], X[:, feat2_index], c=y, alpha=0.5)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(feat1 + ' - ' + feat2)
    plt.show()

    return accuracy_score(y, pred)

def view_data():
    train = pd.read_csv('data/user_example.csv')
    train_labels = train['TARGET']
    print(train.head())

    train = train.drop(['UserID', 'TARGET'], axis=1)

    fs = FeatureSelector(data = train, labels = train_labels)

    fs.identify_missing(missing_threshold=0.6)

    missing_features = fs.ops['missing']
    print(missing_features[:10])

    fs.plot_missing()

    print(fs.missing_stats.head(10))

    fs.identify_single_unique()

    single_unique = fs.ops['single_unique']
    print(single_unique)

    print(fs.unique_stats.sample(5))

    fs.identify_collinear(correlation_threshold=0.975)

    correlated_features = fs.ops['collinear']
    print(correlated_features[:5])

    fs.plot_collinear()
    # fs.plot_collinear(plot_all=True)

    fs.identify_collinear(correlation_threshold=0.98)
    fs.plot_collinear()

    print(fs.record_collinear.head())

    fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                                n_iterations = 10, early_stopping = True)

    one_hot_features = fs.one_hot_features
    base_features = fs.base_features
    print('There are %d original features' % len(base_features))
    print('There are %d one-hot features' % len(one_hot_features))

    print(fs.data_all.head(10))

    zero_importance_features = fs.ops['zero_importance']
    print(zero_importance_features[10:15])

    fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

    print(fs.feature_importances.head(10))

if __name__ == "__main__":
    func_name = args.func
    if func_name == 'view_data':
        view_data()
    elif func_name == 'select_features':
        select_features()
    elif func_name == 'train':
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        train(X_train, y_train)
        acc = validate(X_test, y_test)
        print(acc)
    elif func_name == 'validate':
        X, y, feat_names = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        acc = validate(X_test, y_test, feat_names)
        print(acc)
        