import numpy as np
import pandas as pd
import sklearn.svm


def convert_data_to_dummies(fn, n_columns=None):
    # TODO: check for null features
    df = pd.read_csv(fn)
    labels = df[df.columns[0]]
    labels = pd.get_dummies(labels)
    labels.to_csv('labels.csv')
    features = df[df.columns[1:]]
    if n_columns is not None:
        features = features[features.columns[:n_columns]]
    features_dummies = pd.get_dummies(features)
    features_dummies.to_csv('features.csv')


def train_test_val_split(input_df: pd.DataFrame, output_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    shape = input_df.shape
    r = np.random.rand(shape[0])
    train_selector = r < 0.8
    train = input_df[train_selector]
    test = input_df[~train_selector]
    train_output = output_df[train_selector]
    test_output = output_df[~train_selector]
    return train, test, train_output, test_output


def svm_score(train_X, train_Y, test_X, test_Y, type='rbf', C=0.1):
    model = sklearn.svm.SVC(kernel=type, C=C)
    model.fit(np.array(train_X.tolist()), np.array(train_Y.tolist()))
    return model.score(test_X, test_Y)


def main(type, C):
    # input_ = np.array([[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]])
    # output = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1])
    input_ = pd.read_csv('features.csv', index_col=0).values
    output = pd.read_csv('labels.csv', index_col=0).values[:, 0]
    k = 5
    selector = np.random.rand(output.shape[0])
    delta = 1 / float(k)
    bucket_selectors = []
    for b in range(k):
        bucket_selector = [n for n in range(output.shape[0]) if b * delta < selector[n] <= (b + 1) * delta]
        bucket_selectors.append(bucket_selector)
    input_buckets = np.array([input_[s].tolist() for s in bucket_selectors])
    output_buckets = np.array([output[s].tolist() for s in bucket_selectors])
    scores = []
    for n in np.arange(k):
        test_X, test_Y = input_buckets[n], output_buckets[n]
        indices = np.arange(k)
        indices = np.delete(indices, n)
        train_X, train_Y = np.concatenate(input_buckets[indices]), np.concatenate(output_buckets[indices])
        scores.append(svm_score(train_X, train_Y, test_X, test_Y, type=type, C=C))
    # print(scores)
    return np.array(scores).mean()


if __name__ == '__main__':
    convert_data_to_dummies('mushrooms.csv')
    for type in ['rbf', 'linear', 'poly', 'sigmoid']:
        for C in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            print(type, C, main(type, C))
