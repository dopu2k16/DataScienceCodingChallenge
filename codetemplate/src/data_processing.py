import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    """
    Loading the dataset if the file exists.
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df
    return None


def preprocess_dataset(input_data):
    """ The preprocessing selects the relevant data.

    :param input_data: Input data
    :return X: Transformed data containing the training data.
    :rtype: .... """

    X = input_data
    # Putting response variable to y
    y = X['b_gekauft_gesamt']
    # dropping the target variable for the training data
    X = X.drop('b_gekauft_gesamt', axis=1)
    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        test_size=0.2, random_state=100)

    # Scaling
    scaler = StandardScaler()
    # the fields in the data with binary values despite being int64/float64 datatype
    binary_vals = ['fakeID', 'b_specialisation_i', 'b_specialisation_a',
                   'b_specialisation_b', 'b_specialisation_c', 'b_specialisation_d',
                   'b_specialisation_e', 'b_specialisation_f', 'b_specialisation_g',
                   'b_specialisation_h', 'b_specialisation_j', 'b_in_kontakt_gewesen',
                   'b_gekauft_gesamt']

    # Scaling the dataset with numerical values
    for i in X.columns:
        if i not in binary_vals:
            X_train[[i]] = scaler.fit_transform(X_train[[i]])
            X_test[[i]] = scaler.fit_transform(X_test[[i]])

    return X_train, y_train, X_test, y_test


def main():
    input_data = load_data('../../data/CustomerData_LeadGenerator.csv')
    X_train, y_train, X_test, y_test = preprocess_dataset(input_data)
    X_train.to_csv('X_train.csv')
    X_test.to_csv('y_train.csv')
    X_test.to_csv('X_test.csv')
    y_test.to_csv('y_test.csv')


if __name__ == "__main__":
    main()
