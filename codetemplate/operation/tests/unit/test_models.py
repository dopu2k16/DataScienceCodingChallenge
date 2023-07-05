import unittest

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from codetemplate.src.batch_score import batch_prediction
from codetemplate.src.data_processing import load_data, preprocess_dataset


class ModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setting up the dataset for unit tests.
        """
        filename = \
            '../../../../data/data_unittest.csv'
        self.dataset = load_data(filename)
        # Putting response variable to y
        y = self.dataset['b_gekauft_gesamt']
        # dropping the target variable for the training data
        X = self.dataset.drop('b_gekauft_gesamt', axis=1)
        # Splitting the data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test\
            = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)

    def test_output_shape_of_decision_tree_classifier(self):
        """
        Test to check the output shape of the Decision Tree Classifier.
        """
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)
        pred_train = dt.predict(self.X_train)
        pred_test = dt.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],),\
            'DecisionTree output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],),\
            'DecisionTree output should be same as testing labels.'

    def test_output_shape_of_random_forrest_classifier(self):
        """
        Test to check the output shape of the Random Forrest Classifier.
        """
        rf = RandomForestClassifier()
        rf.fit(self.X_train, self.y_train)
        pred_train = rf.predict(self.X_train)
        pred_test = rf.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],),\
            'RandomForrest output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],),\
            'RandomForrest output should be same as testing labels.'

    def test_output_shape_of_logistic_regression_classifier(self):
        """
        Test to check the output shape of the Logistic Regression Classifier.
        """
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)
        pred_train = lr.predict(self.X_train)
        pred_test = lr.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],),\
            'Logistic Regression output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],),\
            'Logistic Regression output should be same as testing labels.'

    def test_output_shape_of_perceptron_classifier(self):
        """
        Test to check the output shape of the Perceptron Classifier.
        """
        perc = Perceptron()
        perc.fit(self.X_train, self.y_train)
        pred_train = perc.predict(self.X_train)
        pred_test = perc.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],), \
            'Perceptron output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],), \
            'Perceptron output should be same as testing labels.'

    def test_output_shape_of_mlp_classifier(self):
        """
        Test to check the output shape of the Multi-layer Perceptron Classifier.
        """
        mlp = MLPClassifier(random_state=1, early_stopping=True)
        mlp.fit(self.X_train, self.y_train)
        pred_train = mlp.predict(self.X_train)
        pred_test = mlp.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],), \
            'Multi-layer Perceptron output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],), \
            'Multi-layer Perceptron output should be same as testing labels.'

    def test_output_shape_of_gbr_classifier(self):
        """
        Test to check the output shape of the Gradient Boosting Classifier.
        """
        gbr = GradientBoostingClassifier()
        gbr.fit(self.X_train, self.y_train)
        pred_train = gbr.predict(self.X_train)
        pred_test = gbr.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert pred_train.shape == (self.X_train.shape[0],), \
            'Gradient Boosting output should be same as training labels.'
        assert pred_test.shape == (self.X_test.shape[0],), \
            'Gradient Boosting output should be same as testing labels.'

    def test_output_range_of_decision_tree_classifier(self):
        """
        Test to check the output range of the Decision Tree Classifier.
        """
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)
        pred_train = dt.predict(self.X_train)
        pred_test = dt.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'Decision tree output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'Decision tree output should range from 0 to 1 inclusive'

    def test_output_range_of_logistic_regression_classifier(self):
        """
        Test to check the output range of the Logistic Regression Classifier.
        """
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)
        pred_train = lr.predict(self.X_train)
        pred_test = lr.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'Logistic Regression output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'Logistic Regression output should range from 0 to 1 inclusive'

    def test_output_range_of_gbr_classifier(self):
        """
        Test to check the output range of the Gradient Boosting Classifier.
        """
        gbr = GradientBoostingClassifier()
        gbr.fit(self.X_train, self.y_train)
        pred_train = gbr.predict(self.X_train)
        pred_test = gbr.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'Gradient Boosting Regression output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'Gradient Boosting Regression output should range from 0 to 1 inclusive'

    def test_output_range_of_random_forrest_classifier(self):
        """
        Test to check the output range of the Random Forrest Classifier.
        """
        rf = RandomForestClassifier()
        rf.fit(self.X_train, self.y_train)
        pred_train = rf.predict(self.X_train)
        pred_test = rf.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'RandomForrest output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'RandomForrest output should range from 0 to 1 inclusive'

    def test_output_range_of_perceptron_classifier(self):
        """
        Test to check the output range of the Perceptron Classifier.
        """
        pr = Perceptron()
        pr.fit(self.X_train, self.y_train)
        pred_train = pr.predict(self.X_train)
        pred_test = pr.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'Perceptron output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'Perceptron output should range from 0 to 1 inclusive'

    def test_output_range_of_mlp_classifier(self):
        """
        Test to check the output range of the Multi-layer Perceptron Classifier.
        """
        mlp = MLPClassifier(random_state=1, early_stopping=True)
        mlp.fit(self.X_train, self.y_train)
        pred_train = mlp.predict(self.X_train)
        pred_test = mlp.predict(self.X_test)

        # =================================
        # TEST SUITE
        # =================================
        assert (pred_train <= 1).all() & (pred_train >= 0).all(), \
            'Multi-layer Perceptron output should range from 0 to 1 inclusive'
        assert (pred_test <= 1).all() & (pred_test >= 0).all(), \
            'Multi-layer Perceptron output should range from 0 to 1 inclusive'

    def test_model_returns_correct_type_object(self):
        """
        Test for the return of the correct object of the modeling function.
        """
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scores = cross_validate(RandomForestClassifier(), self.X_train, self.y_train,
                                scoring=('accuracy', 'f1_weighted'),
                                cv=cv, n_jobs=-1, return_train_score=True)

        # =================================
        # TEST SUITE
        # =================================
        # Check the return object type
        assert isinstance(scores, dict)
        # Check the length of the returned object
        assert len(scores) == 6
        # Check the correctness of the names of the returned dict keys
        assert 'test_accuracy' in scores and 'test_f1_weighted' in scores
        assert 'train_accuracy' in scores and 'train_f1_weighted' in scores

    def test_model_returns_correct_values(self):
        """
        Tests for the returned values of the modeling function.
        """
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scores = cross_validate(RandomForestClassifier(), self.X_train, self.y_train,
                                scoring=('accuracy', 'f1_weighted'),
                                cv=cv, n_jobs=-1, return_train_score=True)

        # =================================
        # TEST SUITE
        # =================================
        # Check returned scores' type
        print(type(scores['train_accuracy']))
        assert isinstance(scores['train_accuracy'], np.ndarray)
        assert isinstance(scores['test_accuracy'], np.ndarray)
        assert isinstance(scores['train_f1_weighted'], np.ndarray)
        assert isinstance(scores['test_f1_weighted'], np.ndarray)
        # Check returned scores' range
        assert scores['train_accuracy'].mean() >= 0.0
        assert scores['test_accuracy'].mean() <= 1.0
        assert scores['train_f1_weighted'].mean() >= 0.0
        assert scores['test_f1_weighted'].mean() <= 1.0

    def test_wrong_input_raises_assertion(self):
        """
        Tests for various assertion cheks written in the modeling function.
        """
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        scores = cross_validate(RandomForestClassifier(), self.X_train, self.y_train,
                                scoring=('accuracy', 'f1_weighted'),
                                cv=cv, n_jobs=-1, return_train_score=True)

        # =================================
        # TEST SUITE
        # =================================
        # Test that it handles the case of: X is a string
        with self.assertRaises(TypeError) as exception:
            msg = preprocess_dataset('X')
            assert "TypeError: string indices must be integers" in str(exception.value)
            msg = batch_prediction(self.X_train, self.y_train, self.X_test, self.y_test, None)
            assert "ValueError" in str(exception.value)


if __name__ == '__main__':
    unittest.main()
