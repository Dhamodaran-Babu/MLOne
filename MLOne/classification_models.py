import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,accuracy_score,make_scorer,average_precision_score
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

def calculate_roc(estimator,xval,yval,model_name):
    """calculate auc score and plot the roc and auc"""
    probs = estimator.predict_proba(xval)[:,1]
    fpr,tpr,_ = roc_curve(yval,probs)
    random_probs = [0 for i in range(len(yval))]
    p_fpr,p_tpr,_ = roc_curve(yval,random_probs)
    auc_score = roc_auc_score(yval,probs)
    plt.plot(p_fpr, p_tpr, linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='{} (area={})'.format(model_name,round(auc_score,4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC-AUC CURVE")
    plt.legend()
    plt.savefig('ROC-AUC.jpg')
    return auc_score


def evaluate_performance(estimator,xval,yval,model_name):
    """Results are dictionaries which contains the model type(knn,linear SVC,naive Bayes) as keys.
    They contain info on the performance of the estimator under different criteria"""
    ypred = estimator.predict(xval)
    conf_mat = confusion_matrix(yval,ypred)
    accuracy = (conf_mat[0][0]+conf_mat[1][1])/(len(yval))
    auc_score = calculate_roc(estimator,xval,yval,model_name=model_name)
    avg_precision = average_precision_score(yval,ypred,average='weighted')
    model_results = {'accuracy':accuracy,'AUC':auc_score,'weighted_precision':avg_precision}
    """other metrics such as specificity, sensitvity, precision, recall , f1 score,AUC score has to be included.
        AUC score, precision,accuracy are must F1 - score can be included as it is the mean of precision and recall """
    return model_results
    

def fit_knn(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.neighbors import KNeighborsClassifier

    estimator = KNeighborsClassifier(n_jobs=-1)
    params = {'n_neighbors':range(2,4),'metric':['minkowski','euclidean']}
    scoring = {'Precision':'precision','Accuracy':make_scorer(accuracy_score),'AUC':'roc_auc'}
    random_search = RandomizedSearchCV(estimator=estimator,param_distributions=params,scoring=scoring,refit='AUC',
                                        n_jobs=-1,cv=stratified_splitter,random_state=101,return_train_score=True)
    random_search.fit(xtrain,ytrain.flatten())
    estimator = random_search.best_estimator_
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='KNN')
    
    return estimator,model_results

def fit_logistic_reg(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(random_state=101,n_jobs=-1)
    penalty = ['l2','none']
    c=np.geomspace(start=0.1,stop=10,num=3,endpoint=True)
    params = {'penalty':penalty,'C':c}
    scoring = {'Precision':'precision','Accuracy':make_scorer(accuracy_score),'AUC':'roc_auc'}
    random_search = RandomizedSearchCV(estimator=estimator,param_distributions=params,scoring=scoring,refit='AUC',
                                        n_jobs=-1,cv=stratified_splitter,random_state=101,return_train_score=True)
    random_search.fit(xtrain,ytrain.flatten())
    estimator = random_search.best_estimator_
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='Log_reg')

    return estimator, model_results

def fit_svc(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.svm import SVC
    
    estimator = SVC(probability=True, random_state=101)
    params = {'C' : [1,10,100], 'kernel' : ['rbf', 'linear'], 'gamma' : ['scale', 'auto']}
    scoring = {'Precision':'precision','Accuracy':make_scorer(accuracy_score),'AUC':'roc_auc'}
    random_search = RandomizedSearchCV(estimator=estimator,param_distributions=params,scoring=scoring,refit='AUC',
                                        n_jobs=-1,cv=stratified_splitter,random_state=101,return_train_score=True)
    random_search.fit(xtrain,ytrain.flatten())
    estimator = random_search.best_estimator_
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='SVC')
    return estimator,model_results

def fit_random_forest(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.ensemble import RandomForestClassifier
    
    estimator = RandomForestClassifier(random_state=101,n_jobs=-1)
    params = {'n_estimators' : [10,100,200], 'criterion' : ['gini', 'entropy']}
    scoring = {'Precision':'precision','Accuracy':make_scorer(accuracy_score),'AUC':'roc_auc'}
    random_search = RandomizedSearchCV(estimator=estimator,param_distributions=params,scoring=scoring,refit='AUC',
                                        n_jobs=-1,cv=stratified_splitter,random_state=101,return_train_score=True)
    random_search.fit(xtrain,ytrain.flatten())
    estimator = random_search.best_estimator_
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='Random_Forest')
    return estimator,model_results

def fit_decision_tree(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.tree import DecisionTreeClassifier
    
    estimator = DecisionTreeClassifier(random_state=101)
    params = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}
    scoring = {'Precision':'precision','Accuracy':make_scorer(accuracy_score),'AUC':'roc_auc'}
    random_search = RandomizedSearchCV(estimator=estimator,param_distributions=params,scoring=scoring,refit='AUC',
                                        n_jobs=-1,cv=stratified_splitter,random_state=101,return_train_score=True)
    random_search.fit(xtrain,ytrain.flatten())
    estimator = random_search.best_estimator_
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='Decision_Tree')
    return estimator,model_results

def fit_GaussianNB(xtrain,xval,ytrain,yval,stratified_splitter):
    from sklearn.naive_bayes import GaussianNB
    
    estimator = GaussianNB()
    estimator.fit(xtrain,ytrain.flatten())
    model_results = evaluate_performance(estimator=estimator,xval=xval,yval=yval,model_name='GaussianNB')
    return estimator,model_results
    


def model_fitter(xtrain,ytrain,xval,yval):
    print("\n\n<<<<FITTING CLASSIFICATION MODELS>>>>\n\n")
    from sklearn.model_selection import StratifiedShuffleSplit
    stratified_splitter = StratifiedShuffleSplit(test_size=0.2,random_state=101)

    """Results are dictionaries which contains the model type(knn,linear SVC,naive Bayes) as keys.
    They contain info on the performance of the estimator under different criteria"""
    results = {}

    all_estimators={}
    
    all_estimators['KNN'],results['KNN'] = fit_knn(xtrain,xval,ytrain,yval,stratified_splitter)
    if len(np.unique(yval)) <= 2:
        all_estimators['Log_reg'],results['Log_reg'] = fit_logistic_reg(xtrain,xval,ytrain,yval,stratified_splitter)
    all_estimators['SVC'],results['SVC'] = fit_svc(xtrain,xval,ytrain,yval,stratified_splitter)
    all_estimators['Random_Forest'],results['Random_Forest'] = fit_random_forest(xtrain,xval,ytrain,yval,stratified_splitter)
    all_estimators['Decision_Tree'],results['Decision_Tree'] = fit_decision_tree(xtrain,xval,ytrain,yval,stratified_splitter)
    all_estimators['GaussianNB'],results['GaussianNB'] = fit_GaussianNB(xtrain,xval,ytrain,yval,stratified_splitter)
    
    return all_estimators,results