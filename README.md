This package enables you to directly fit the best Machine Learning Model for your dataset by automating all the preprocessing and model fitting steps, Additionally it also performs Exploratory data analysis in the dataset.

Code Snippet:
Auto_Fit(File_Path, bias_var=False)

Parameter:
bias_var: {True, False}(Default: False)
Calculates average Bias, Variance and Expected Loss for all the models.


Rules and guide lines for uploading the dataset:
1. The file should be either .csv or .xlsx
2. Number of columns : 3 < cols > 100
3. Number of rows : 200 < rows > 2500
4. The index col must be the first column.If the dataset doesn't have an index column include it.For example,you can use row number as index.
5. The dependent variable or the target class should be the last column

Model default settings:
chi square Test
p val < 0.1 

Train Test Validation split ratio ** 70:20:10
SSS No.of folds ** 10

Random search params
scores = AUC,precision,accuracy
refit criterion = AUC

KNN params:
2 < n_neighbors < 5
metric = euclidean,manhattan,minkowski

Logistic Regression:
penalty = l1,none
solver = default
c = 0.1 geomspace,no.of elements =3

SVC
params = {'C' : [1,10,100], 'kernel' : ['rbf', 'linear'], 'gamma' : ['scale', 'auto']}

Random Forest Classifier
params = {'n_estimators' : [10,100,200], 'criterion' : ['gini', 'entropy']}

Decision Trees
params = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}

Naive Bayes(Gaussian)
default parameters