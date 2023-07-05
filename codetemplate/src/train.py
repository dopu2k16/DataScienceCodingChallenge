from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Perceptron, PoissonRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from codetemplate.src.batch_score import batch_prediction
from codetemplate.src.data_processing import preprocess_dataset, load_data

import warnings
warnings.filterwarnings('ignore')


def get_ml_models():
    """
    The ML training algorithms for the Lead Generator Problem.
    The following ml algorithms were used to predict a given customer as a hot lead or not.
    """
    models = dict()
    # Logistic regression
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
    model = LogisticRegression()
    models['LR'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    # Perceptron
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=Perceptron(), n_features_to_select=10)
    model = Perceptron()
    models['Perceptron'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    # Decision Tree
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
    model = DecisionTreeClassifier()
    models['CART'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    # Random Forest
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
    model = RandomForestClassifier()
    models['RandomForest'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    # Gradient Boosting
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=10)
    model = GradientBoostingClassifier()
    models['GBM'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    # Multilayer Perceptron
    # feature selection by Recursive Feature Elimination
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
    model = MLPClassifier(random_state=1, early_stopping=True)
    models['MLP'] = Pipeline(steps=[('feature_selection', rfe), ('m', model)])

    return models


def main():
    input_data = load_data('../../data/CustomerData_LeadGenerator.csv')
    X_train, y_train, X_test, y_test = preprocess_dataset(input_data)
    # getting all the implemented ml models
    models = get_ml_models()
    # getting the predictions for both the training and testing datasets
    batch_prediction(X_train, y_train, X_test, y_test, models)


if __name__ == "__main__":
    main()
