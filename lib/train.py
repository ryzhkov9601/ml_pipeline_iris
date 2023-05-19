import json
import os
import pickle
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn import datasets
from sklearn.model_selection import train_test_split
# different classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
#
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import mlflow

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('ryzhkov9601')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}

# {"command": "dvc exp run -S train.model=knn"}
# {"command": "dvc exp run -S train.model=svm_linear"}
# {"command": "dvc exp run -S train.model=svm_rbf"}
# {"command": "dvc exp run -S train.model=gaussian_process"}
# {"command": "dvc exp run -S train.model=decision_tree"}
# {"command": "dvc exp run -S train.model=random_forest"}
# {"command": "dvc exp run -S train.model=neural_net"}
# {"command": "dvc exp run -S train.model=adaboost"}
# {"command": "dvc exp run -S train.model=gaussian_nb"}
# {"command": "dvc exp run -S train.model=qda"}
# {"command": "dvc exp run -S train.model=logistic_regression"}
MODELS = {
    'knn': KNeighborsClassifier(3),
    'svm_linear': SVC(kernel="linear", C=0.025),
    'svm_rbf': SVC(gamma=2, C=1),
    'gaussian_process': GaussianProcessClassifier(1.0 * RBF(1.0)),
    'decision_tree': DecisionTreeClassifier(max_depth=5),
    'random_forest': RandomForestClassifier(max_depth=5, n_estimators=100, max_features=2),
    'neural_net': MLPClassifier(alpha=1, max_iter=1000),
    'adaboost': AdaBoostClassifier(),
    'gaussian_nb': GaussianNB(),
    'qda': QuadraticDiscriminantAnalysis(),
    'logistic_regression': LogisticRegression(),
}


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(x, y, model_name):
    model = MODELS[model_name]
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']

    task_dir = 'data/train'

    data = load_dict('data/preprocessing/data.json')

    model = train_model(data['train_x'], data['train_y'], config['model'])

    preds = model.predict(data['train_x'])

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['train_y'], preds)

    report = classification_report(data['train_y'], preds, output_dict=True)

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))
    save_dict(report, os.path.join(task_dir, 'report.json'))

    sns.heatmap(pd.DataFrame(data['train_x']).corr())

    plt.savefig('data/train/heatmap.png')

    with open('data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact('data/train/report.json')
    mlflow.log_artifact('data/train/heatmap.png')
    mlflow.sklearn.log_model(model, 'model.pkl')


if __name__ == '__main__':
    train()
