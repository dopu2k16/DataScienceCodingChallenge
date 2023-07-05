# Lead Generator using Machine Learning

## Author: Mitodru Niyogi

The lead generator is a binary class problem to determine whether a given customer is a lead or not.

Feature selection was performed by Recursive Feature Elimination and models like Decision Tree, Random Forest, Perceptron, GradientBoosting.

Among all the models, Gradient Boosting model performs the best and has achieved the highest test accuracy of 99.338\% and 0.992 test F1-score.

The notebook [codetemplate/notebooks/EDA.ipynb](codetemplate/notebooks/EDA.ipynb) contains the data exploration and analysis. The modeling, experiments, training,
 and evaluation of the algorithms are explained in the [codetemplate/notebooks/Modeling-Training-Evaluation.ipynb](codetemplate/notebooks/Modeling-Training-Evaluation.ipynb) notebook.
 
[codetemplate](codetemplate) directory contains the source files and the unittests and data validation tests for the project. 

[codetemaplate/src](codetemplate/src) contains the source code for the tasks.
[codetemplate/results](codetemplate/results) contains the result files.
[(codetemplate/operation/tests/](codetemplate/operation/tests/) contains unittests for model and data validation.

[codetemplate/src/data_processing.py](codetemplate/src/data_processing.py) contains the preprocessing methods for spliting the dataset into train, validation, and test sets, and scaling the numerical features.


[codetemplate/src/models.py](codetemplate/src/models.py) has machine learning models implementation for classification.

[codetemplate/src/batch_score.py](codetemplate/src/batch_score.py) has classifier prediction methods.

[codetemplate/src/evaluate.py](codetemplate/src/evaluate.py) has evaluation methods.

[codetemplate/src/train.py](codetemplate/src/train.py) contains the training code for various machine learning algorithms for the classification task.

[unittests](codetemplate/operation/tests/unit/test_models.py)
The unitests test the model output shape and output range of each classifier model. 

[codetemplate/operation/tests/data_validation/test_data.py](codetemplate/operation/tests/data_validation/test_data.py) contains tests for data validation such as  load_data_calls_read_csv_if_exists, if_no_nan_values_exists, if_no_null_values_exists, if_duplicates_exists, data_exits_for_all_id, if_target_has_binary_data, if_train_and_test_matrix_have_same_dimension.
