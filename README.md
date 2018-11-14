## Determine Forest Cover Type Based on Only Cartographic Variables Using `RandomForestClassifier`
### Data Info Summary
Dataset and full description can be found [here](https://archive.ics.uci.edu/ml/datasets/Covertypez).
* Contains 581,012 samples
* 54 cartographic attributes
    * 10 numerical variables
    * 44 binary columns (4 wilderness area, 40 soil type)
* 7 different forest cover types
### Data Exploration
It was immediately clear that the 7 forest cover types are not equally represented in the dataset. A 'standard' `train_test_split` would therefore not be sufficient. By using `StratifiedShuffleSplit` in conjunction with `train_test_split` it was possible to build a balanced training set as well as validation and testing sets with class distributions approximately equal to that of the original dataset.
* Training set - 9,611 samples (balanced)
* Validation set - 57,140 samples
* Testing set - 514,261 samples

A `RandomForestClassifier` was then trained and validated using all available features, the top 10 most important features, and the top 20 most important features. The results using all features and the top 20 features is shown below. The results using the top 10 features are excluded because they were not explored any further.

#### Baseline results using all features
Accuracy: 0.6923

Classification Report

|cover type|precision|recall|f1-score|support|
|---|---|---|---|---|
|1|0.69|0.75|0.72|21047|
|2|0.83|0.61|0.70|28193|
|3|0.74|0.77|0.76|3438|
|4|0.32|0.97|0.48|137|
|5|0.20|0.91|0.33|812|
|6|0.48|0.77|0.59|1599|
|7|0.53|0.95|0.68|1914|
|avg / total|0.75|0.69|0.70|57140|

#### Baseline results using top 20 features
Accuracy: 0.6759

Classification Report

|cover type|precision|recall|f1-score|support|
|---|---|---|---|---|
|1|0.68|0.74|0.71|21047|
|2|0.83|0.58|0.68|28193|
|3|0.73|0.78|0.75|3438|
|4|0.33|0.96|0.49|137|
|5|0.19|0.92|0.32|812|
|6|0.45|0.76|0.56|1599|
|7|0.49|0.95|0.65|1914|
|avg / total|0.74|0.68|0.69|57140|

### Feature Engineering
Feature engineering was guided by the results of visualizing the relationships between certain variables and generated 6 new features. All engineered features made it into the top 20 most important features.

#### Results using engineered features and all original features
Accuracy: 0.7148

Classification Report

|cover type|precision|recall|f1-score|support|
|---|---|---|---|---|
|1|0.70|0.77|0.74|21047|
|2|0.85|0.63|0.72|28193|
|3|0.76|0.83|0.79|3438|
|4|0.39|0.97|0.56|137|
|5|0.23|0.93|0.37|812|
|6|0.52|0.82|0.64|1599|
|7|0.54|0.95|0.69|1914|
|avg / total|0.76|0.71|0.72|57140|

#### Results using only the top 20 features (original + engineered)
Accuracy: 0.7079

Classification Report

|cover type|precision|recall|f1-score|support|
|---|---|---|---|---|
|1|0.70|0.75|0.73|21047|
|2|0.85|0.63|0.72|28193|
|3|0.76|0.82|0.79|3438|
|4|0.41|0.98|0.58|137|
|5|0.22|0.94|0.36|812|
|6|0.52|0.82|0.64|1599|
|7|0.49|0.97|0.65|1914|
|avg / total|0.76|0.71|0.72|57140|

### Hyperparameter Optimization
As can be seen above, the results obtained using all features and the top 20 features are very similar. Therefore only the top 20 features were used during hyperparameter optimization. The whole purpose of engineering features was to incease the classification abilities of the model while using fewer dimensions.

Due to the size of this dataset, it is possible to tune hyperparameters based on the performance of the classifier on a validation set that is representative of the unseen test set. `GridSearchCV` uses cross validation on the provided training set to determine the optimal hyperparameters and does not have support for providing a custom validation set. Enter [hypopt](https://pypi.org/project/hypopt/), which offers exactly that functionality. Description from the link: "Grid search hyper-parameter optimization using a validation set (not cross validation)"

The hyperparameters of `RandomForestClassifier` were thoroughly explored (a more detailed description of this process can be found in the notebook). The optimal set of hyperparameters for this particular dataset: `n_estimators = 406`, `bootstrap = False` and the rest remain default.

#### Results using the tuned classifier on the test set

Accuracy of optimized model: 0.7598

Classification Report

|cover type|precision|recall|f1-score|support|
|---|---|---|---|---|
|1|0.76|0.79|0.77|189420|
|2|0.88|0.70|0.78|253735|
|3|0.80|0.85|0.82|30943|
|4|0.46|0.97|0.62|1237|
|5|0.27|0.96|0.42|7308|
|6|0.56|0.90|0.69|14395|
|7|0.53|0.98|0.69|17223|
|avg / total|0.80|0.76|0.77|514261|

### Ideas for future improvements/testing
* Use PCA to generate new features and select top PCs to use for classification
* Test different classifiers in an attempt to improve precision scores (this is not possible with the current classifier, see the notebook)
