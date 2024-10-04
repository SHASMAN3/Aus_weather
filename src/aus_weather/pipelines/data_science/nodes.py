import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import sys
import warnings
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def split_feature_target(df,target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_logistic_regression_model(X_train, X_test, y_train, y_test):
    y_train= y_train.values.ravel()
    y_test= y_test.values.ravel()
    
    logreg = LogisticRegression(solver='liblinear',random_state=0)
    logreg.fit(X_train, y_train)

    y_pred_train = logreg.predict(X_train)
    # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    y_pred_test = logreg.predict(X_test)
    
    y_pred_train_df=pd.DataFrame(y_pred_train,columns=['prediction'])
    y_pred_test_df=pd.DataFrame(y_pred_test,columns=['prediction'])
    
    return y_pred_train_df,y_pred_test_df, logreg

# def evaluate_model(y_pred_train,y_pred_test,y_train,y_test):
#     results= {}
    
#     train_accuracy= accuracy_score(y_train,y_pred_train)
#     test_accuracy= accuracy_score(y_test,y_pred_test)
    
#     print('Accuracy of logistic regression classifier on train set: {:.4f}'.format(train_accuracy))
#     print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(test_accuracy))

#     results['train_accuracy']=train_accuracy
#     results['test_accuracy']=test_accuracy
    
#     # classification report on test set
#     print("classificaton report: \n",classification_report(y_test,y_pred_test))
#     results['classification_report'] = classification_report(y_test,y_pred_test,output_dict=True)
    
#     cm= confusion_matrix(y_test,y_pred_test)
#     print('confusion_matrix\n\n',cm)
#     results['confusion_matrix']=cm.tolist()
    
#     tp=cm[0][0]
#     tn=cm[1][1]
#     fp=cm[0][1]
#     fn=cm[1][0]
    
#     print('\nTrue Positive(TP)  = ', tp)
#     print('False Positive(FP) = ', tn)
#     print('True Negative(TN)  = ', fp)
#     print('False Negative(FN) = ', fn)
    
#     # classification accuracy
#     classification_accuracy = (tp + tn) / (tp + fp + tn + fn)
#     print('\nClassification accuracy = ', classification_accuracy)
#     # classification error
#     classification_error = (fp + fn) / (tp + fp + tn + fn)
#     print('Classification error = ', classification_error)
#     # precision
#     precision = tp / (tp + fp)
#     print('Precision = ', precision)
#     # recall/sensitivity
#     recall = tp / (tp + fn)
#     print('Recall = ', recall)
#     True_Positive_Rate = tp / (tp + fn)
#     print('True Positive Rate = ', True_Positive_Rate)
#     False_Positive_Rate= fp/ (fp+tn)
#     print('False_Positive_Rate=', False_Positive_Rate)
#     # specificity
#     specificity = tn / (tn + fp)
#     print('Specificity = ', Specificity)
    
#     results.update({
#         'classification_accuracy': classification_accuracy,
#         'classification_error': classification_error,
#         'precision': precision,
#         'recall': recall,
#         'True_Positive_Rate': True_Positive_Rate,
#         'False_Positive_Rate': False_Positive_Rate,
#         'Specificity': Specificity
#     })
    
#     return results


def evaluate_model(y_pred_train,y_pred_test,y_train,y_test):
    results= {}
    
    train_accuracy= accuracy_score(y_train,y_pred_train)
    test_accuracy= accuracy_score(y_test,y_pred_test)
    
    print('Accuracy of logistic regression classifier on train set: {:.4f}'.format(train_accuracy))
    print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(test_accuracy))

    results['train_accuracy']=train_accuracy
    results['test_accuracy']=test_accuracy
    
    # classification report on test set
    print("classificaton report: \n",classification_report(y_test,y_pred_test))
    classification_rep = classification_report(y_test,y_pred_test,output_dict=True)
    
    cm= confusion_matrix(y_test,y_pred_test)
    print('confusion_matrix\n\n',cm)
    results['confusion_matrix']=cm.tolist()
    
    tp=cm[0][0]
    tn=cm[1][1]
    fp=cm[0][1]
    fn=cm[1][0]
    
    print('\nTrue Positive(TP)  = ', tp)
    print('False Positive(FP) = ', tn)
    print('True Negative(TN)  = ', fp)
    print('False Negative(FN) = ', fn)
    
    # classification accuracy
    classification_accuracy = (tp + tn) / (tp + fp + tn + fn)
    print('\nClassification accuracy = ', classification_accuracy)
    # classification error
    classification_error = (fp + fn) / (tp + fp + tn + fn)
    print('Classification error = ', classification_error)
    # precision
    precision = tp / (tp + fp)
    print('Precision = ', precision)
    # recall/sensitivity
    recall = tp / (tp + fn)
    print('Recall = ', recall)
    True_Positive_Rate = tp / (tp + fn)
    print('True Positive Rate = ', True_Positive_Rate)
    False_Positive_Rate= fp/ (fp+tn)
    print('False_Positive_Rate=', False_Positive_Rate)
    # specificity
    specificity = tn / (tn + fp)
    print('specificity = ', specificity)
    
    results.update({
        'classification_accuracy': classification_accuracy,
        'classification_error': classification_error,
        'precision': precision,
        'recall': recall,
        'True_Positive_Rate': True_Positive_Rate,
        'False_Positive_Rate': False_Positive_Rate,
        'Specificity': specificity
    })
    
    for key, value in classification_rep.items():
        if isinstance(value,dict):
            for sub_key, sub_value in value.items():
                results[f'{key}_{sub_key}']= sub_value
        else:
            results[key]=value
            
    results_df= pd.DataFrame(list(results.items()),columns=['metric','value'])
    return results_df

