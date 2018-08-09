import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report,precision_recall_fscore_support
import keras
from keras.callbacks import EarlyStopping, Callback
from keras.models import Sequential
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Activation, merge
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline
np.random.seed(10)

def load_data(csv,drop_columns=None,target_column=None):
    """
    Takes a .csv file, drops the columns specified and separates the features
    and target into arrays

    Parameters
    ----------
    csv: string, name of file
    drop_columns: list of strings, default = None, column labels to drop
    target_column: list of strings, default = None, label or target to predict
    in supervised learning

    Returns
    -------
    X: array-like, features
    Y: array-like, target

    """
    df=pd.read_csv(csv)
    nan_rows = df[df.isnull().T.any().T]
    print('Removing '+str(nan_rows.shape[0]) +' nan rows')
    df=df.dropna()
    print('Original dataset shape: '+ str(df.shape))
    if drop_columns==None:
        pass
    else:
        df=df.drop(drop_columns,axis=1)
    if target_column==None:
        X=np.array(df)
        print('Features shape after dropping formulaA and formulaB: '+str(X.shape))
        return X
    else:
        Y=df[target_column]
        Y=np.array(Y)
        df=df.drop([target_column],axis=1)
        X=np.array(df)
        print('Features shape after dropping formulaA,formulaB and StabilityVector: '+str(X.shape))
        return X,Y

def output_string_to_float(Y):
    """
    Transforms the stability vector from an array of strings to an array of floats

    Parameters
    ----------
    Y: array-like, strings

    Returns
    -------
    new_Y: array-like, floats

    """
    new_Y=[]
    #transforming the string to float
    for i in range(len(Y)):
        Y[i] = Y[i][1:-1]
        Y[i] = Y[i].replace(',','')
        Y[i] = Y[i].replace('.0','')
        new_Y.append([float(j) for j in list(Y[i])])
    #dropping the first and last column as they are constant everywhere
    new_Y = np.array(new_Y)[:,1:-1]
    print('Target shape: '+str(new_Y.shape))
    return new_Y

def data_split(features,target):
    """
    Splits the features and target arrays into training and testing sets
    of proportions 0.75 and 0.25 respectively

    Parameters
    ----------
    features: array-like
    target: array-like

    Returns
    -------
    X_train: array-like, training set of features
    y_train: array-like, training set of target
    X_test: array-like, testing set of features
    y_test: array-like, testing set of target

    """
    X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=10)
    print(str(25)+'% Data split into test set')
    print('X_train shape: '+str(X_train.shape))
    print('y_train shape: '+str(y_train.shape))
    print('X_test shape: '+str(X_test.shape))
    print('y_test shape: '+str(y_test.shape))
    return X_train, X_test, y_train, y_test

def normalize_data(features_train,features_test):
    """
    Standardizes features by removing the mean and scaling to unit variance

    Parameters
    ----------
    features_train: array-like, features used for training the model
    features_test: array-like, features used for testing the model

    Returns
    -------
    X_trained: array-like, transformed features for training the model
    X_tested: array-like, transformed features for testing the model

    """
    scaler=StandardScaler()
    X_trained=scaler.fit_transform(features_train)
    X_tested=scaler.transform(features_test)
    return X_trained,X_tested

def choose_model(ml_model):
    """
    Gives selected hyper parameters and their data type for training models of Random Forest,
    Support Vector Machine, Logistic Regression and Multi-layer Perceptron

    Parameters
    ----------
    ml_model: string, classifier name as called in sklearn

    Returns
    -------
    random_grid: list of strings, list of hyperparameters
    param_type: list of strings, list of respective data type of the hyperparameter

    """
    if ml_model == 'RandomForestClassifier':
        random_grid = ['n_estimators','criterion','max_features','max_depth']
        param_type = ['int','str','int','int']
    elif ml_model == 'SVC':
        random_grid = ['kernel','C','class_weight']
        param_type = ['str','float','dict']
    elif ml_model == 'LogisticRegression':
        random_grid = ['penalty','C','fit_intercept']
        param_type = ['str','float','bool']
    elif ml_model =='MLPClassifier':
        random_grid = ['mlp_units','optimizer','dropout']
        param_type = ['int','str','float']
    else:
        pass
    print('There are '+str(len(random_grid))+' hyper parameters '+str(random_grid)+' of type '+str(param_type)+ ' respectively')
    return random_grid,param_type

def setup_param_grid(x,l):
    """
    Creates a dictionary with user specified values for specific hyperparameters

    Parameters:
    x: list of strings, hyper parameters
    l: list of strings, user specified values for hyper parameters

    Returns:
    new_dict: dictionary of hyper parameters as keys and values for each

    """
    new_dict={}
    for i in range(len(x)):
        for j in range(len(l)):
            new_dict[str(x[i])]=l[i]
    return new_dict

def grid_search(model,X_train,y_train,parameters):
    """
    Fits the Random Forest, Support Vector Machine or Logistic Regression
    classifier to the highest scoring combination of its hyperparameters

    Parameters
    ----------
    model: string, classifier name as called in sklearn
    X_train: array-like
    y_train: array-like
    parameters: dict, user specified values for specific hyperparameters

    Returns
    -------
    clf: object, trained classifier

    """
    scores=[]
    params=[]
    #for random forest
    if model == 'RandomForestClassifier':
        for j in parameters['n_estimators']:
            for k in parameters['criterion']:
                for l in parameters['max_features']:
                    for m in parameters['max_depth']:
                        clf = RandomForestClassifier(n_estimators=j,criterion=k,max_features=l,max_depth=m)
                        params.append([j,k,l,m]) #appends unique combinations of params
                        scores.append(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))#scores the model
        best_params = params[np.argmax(scores)]
        #fits final model on the best params
        clf = RandomForestClassifier(n_estimators=best_params[0],criterion=best_params[1],max_features=best_params[2],
                                     max_depth=best_params[3],random_state=10)
        clf.fit(X_train,y_train)
    #same procedure for SVM
    elif model == 'SVC':
        for j in parameters['kernel']:
            for k in parameters['C']:
                for l in parameters['class_weight']:
                    clf = SVC(kernel=j,C=k,class_weight=l)
                    params.append([j,k,l])
                    scores.append(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))

        best_params = params[np.argmax(scores)]
        clf = SVC(kernel=best_params[0],C=best_params[1],class_weight=best_params[2],random_state=10)
        clf.fit(X_train,y_train)
    #same procedure for logistic regression
    elif model == 'LogisticRegression':
        for j in parameters['penalty']:
            for k in parameters['C']:
                for l in parameters['fit_intercept']:
                    clf = LogisticRegression(penalty=j,C=k,fit_intercept=l)
                    params.append([j,k,l])
                    scores.append(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))

        best_params = params[np.argmax(scores)]
        clf = LogisticRegression(penalty=best_params[0],C=best_params[1],fit_intercept=best_params[2],random_state=10)
        clf.fit(X_train,y_train)

    return clf

def fit_ml_model(X,Y,ml_model,parameters,X_real_test=None):
    """
    Trains the Random Forest, Support Vector Machine or Logistic Regression classifier with
    the highest scoring combination of its hyperparameters and returns its precision, recall and
    f1-score for one label at a time in the target array

    Parameters:
    X: array-like, features
    Y: array-like, targets
    ml_model: string, classifier name as called in sklearn
    parameters: dict, user specified values for specific hyperparameters
    X_real_test: array-like, default=None, testing features

    Returns:
    out_df: dataframe, metrics for each label
    y_real_pred: array-like, returned only when X_real_test!=None, predicted testing target

    """
    d = RandomOverSampler() #initializing oversampling
    X_train, X_test, y_train, y_test = data_split(X,Y) #splitting the data into train and test
    X_res, y_res = d.fit_sample(X_train, y_train)
    print('Resampled X_train shape: '+str(X_res.shape))
    print('Resampled y_train shape: '+str(y_res.shape))
    X_res,X_test = normalize_data(X_res,X_test)

    clf = grid_search(ml_model,X_res,y_res,parameters)
    print(clf)

    y_pred = clf.predict(X_test) # predicting on test set

    clf_rep = precision_recall_fscore_support(y_test, y_pred)
    out_dict = {
             "precision" :clf_rep[0].round(2)
            ,"recall" : clf_rep[1].round(2)
            ,"f1-score" : clf_rep[2].round(2)
            ,"support" : clf_rep[3]
            }
    out_df = pd.DataFrame.from_dict(out_dict) #stores all metrics in a dataframe

    if type(X_real_test) == np.ndarray:
        y_real_pred = clf.predict(X_real_test) #predicts on held out dataset
        return out_df,y_real_pred
    else:
        pass
        return out_df

def metrics_ml_model(X,Y,ml_model,params,X_real_test=None):
    """
    Trains the Random Forest, Support Vector Machine or Logistic Regression classifier with
    the highest scoring combination of its hyperparameters and returns its precision, recall
    and f1-score for all labels in the target array

    Parameters
    ----------
    X: array-like, features
    Y: array-like, targets
    ml_model: string, classifier name as called in sklearn
    parameters: dict, user specified values for specific hyperparameters
    X_real_test: array-like, default=None, testing features

    Returns
    -------
    metrics_df: dataframe, metrics of all labels
    y_real_pred_all: array-like, returned only when X_real_test!=None, predicted testing target for all labels

    """
    metrics_df=[]
    y_real_pred_all=[]
    if type(X_real_test) == np.ndarray:
        for i in range(Y.shape[1]): #fitting classifiers for all labels
            print('Label: '+str(i))
            df1,df2= fit_ml_model(X,Y[:,i],ml_model,params,X_real_test) #calling the fit_ml_model
            metrics_df.append(df1)
            y_real_pred_all.append(df2)
        metrics_df =pd.concat(metrics_df)
        return metrics_df,y_real_pred_all #returns the predictions on the held out set

    else:
        for i in range(Y.shape[1]):
            print('Label: '+str(i))
            df1= fit_ml_model(X,Y[:,i],ml_model,params,X_real_test)
            metrics_df.append(df1)
        metrics_df =pd.concat(metrics_df)
        return metrics_df

def avg_metric(metrics_df,metric):
    """
    Calculates the average metric weighted by the support for each label

    Parameters
    ----------
    metrics_df: dataframe, metrics of all labels
    metric: string

    Returns
    -------
    avg_scores: list, average metrics for each label

    """
    f1= metrics_df[str(metric)]*metrics_df['support'] #metric times the support for each class
    avg_scores=[]
    for i in range(0,len(f1),2):
        weighed = f1.iloc[i:i+2]
        weighed_sum = weighed[0]+weighed[1]
        total = metrics_df['support'][i:i+2].sum()
        avg_scores.append(float(format(weighed_sum/total, '.2f')))#average weighted with support
    return avg_scores

def overall_avg(avg_scores,metric):
    """
    Calculates overall average of the metrics-precision,recall,f1-score

    Parameters
    ----------
    avg_scores: list, average metrics for each label

    Returns
    -------
    overall_avg: float, overall average metric of all labels

    """
    print('The average '+metric+' is: '+format(np.mean(avg_scores), '.2f'))
    return float(format(np.mean(avg_scores), '.2f'))

def fit_mlp_nn(X,Y,j,k,l,X_real_test=None):
    """
    Trains a multi-layer perceptron for one label at a time in the target array and returns its precision, recall
    and f1-score

    Parameters
    ----------
    X: array-like, features
    Y: array-like, target
    j: string, optimizer
    k: float, dropout
    l: int, neurons in each dense layer
    X_real_test: array-like, default=None, testing features

    Returns
    -------
    out_df: dataframe, metrics for each label
    y_real_pred: array-like, returned only when X_real_test!=None, predicted testing target for all labels

    """
    np.random.seed(10)
    d = RandomOverSampler()
    X_train, X_test, y_train, y_test = data_split(X,Y)
    X_res, y_res = d.fit_sample(X_train, y_train)
    X_res, X_test = normalize_data(X_res,X_test)
    print('Resampled X_train shape: '+str(X_res.shape))
    print('Resampled y_train shape: '+str(y_res.shape))
    #the mlp has 3 dense layers with relu activation

    mlp_input = Input(shape=(int(X_res.shape[1]),))
    x = Dense(l,kernel_initializer='glorot_normal', activation="relu")(mlp_input)
    x = Dropout(k)(x)
    x = Dense(l,kernel_initializer='glorot_normal',activation="relu")(x)
    x = Dropout(k)(x)
    x = Dense(l,kernel_initializer='glorot_normal',activation="relu")(x)
    x = Dropout(k)(x)
    x = Dense(1, activation='sigmoid')(x) #classification neuron
    model = Model(mlp_input, x)

    #print(model.summary())
    early = EarlyStopping(monitor='loss', patience=30, verbose=1) #callback to prevent overfitting
    model.compile(optimizer=j, loss="binary_crossentropy",metrics=['accuracy'])
    model.fit(X_res, y_res, epochs=300, batch_size=50,callbacks=[early],verbose=0)

    preds = model.predict(X_test)
    preds[preds>=0.5] = 1 #assigning class 1
    preds[preds<0.5] = 0 #assigning class 0

    clf_rep = precision_recall_fscore_support(y_test, preds)
    out_dict = {
             "precision" :clf_rep[0].round(2)
            ,"recall" : clf_rep[1].round(2)
            ,"f1-score" : clf_rep[2].round(2)
            ,"support" : clf_rep[3]
            }
    out_df = pd.DataFrame.from_dict(out_dict) #stores all metrics in a dataframe

    if type(X_real_test) == np.ndarray:
        y_real_pred = model.predict(X_real_test) #predicts on held out dataset
        return out_df,y_real_pred
    else:
        pass
        return out_df

    return out_df

def metrics_mlp_model(params,X,Y,j,k,l,X_real_test=None):
    """
    Trains a multi-layer perceptron for all labels in the target array and returns its precision, recall
    and f1-score

    Parameters
    ----------
    params: dict, user specified values for specific hyperparameters
    X: array-like, features
    Y: array-like, target
    j: string, optimizer
    k: float, dropout
    l: int, neurons in each dense layer

    Returns
    -------
    metrics_df: dataframe, metrics for all labels
    y_real_pred_all: array-like, returned only when X_real_test!=None, predicted testing target for all labels

    """
    metrics_df=[]
    y_real_pred_all=[]
    if type(X_real_test) == np.ndarray:
        for i in range(Y.shape[1]): #fitting classifiers for all labels
            print('Label: '+str(i))
            df1,df2= fit_mlp_nn(X,Y[:,i],j,k,l,X_real_test) #calling the fit_mlp_nn function
            metrics_df.append(df1)
            y_real_pred_all.append(df2)
        metrics_df =pd.concat(metrics_df)
        return metrics_df,y_real_pred_all
    else:
        for i in range(Y.shape[1]):
            print('Label: '+str(i))
            df1= fit_mlp_nn(X,Y[:,i],j,k,l,X_real_test=None)
            metrics_df.append(df1)
        metrics_df =pd.concat(metrics_df)
        return metrics_df

def wrapper_nn(parameters,X,Y):
    """
    Wraps the mlp functions, calculates metrics for all combinations of hyperparameters for
    the multi-layer perceptron

    Parameters
    ----------
    parameters: dict, user specified values for specific hyperparameters
    X: array-like, features
    Y: array-like, targets

    Returns
    -------
    df3: dataframe, combination of hyperparameters and metrics

    """
    params=[]
    metrics_all_nn=[]
    for j in parameters['optimizer']:
        for k in parameters['dropout']:
            for l in parameters['mlp_units']:
                print('For params: '+str([j,k,l]))
                params.append([j,k,l])
                metrics_df = metrics_mlp_model(parameters,X,Y,j,k,l) #calling metrics_mlp_model
                #print('The metrics for '+str([j,k,l])+str(metrics_df))
                precisions = avg_metric(metrics_df,'precision') #precision values weighted by the support for all 9 labels
                print('The weighted precision for each label: '+str(precisions))
                overall_prec = overall_avg(precisions,'precision')  #average precision
                recall = avg_metric(metrics_df,'recall') #recall values weighted by the support for all 9 labels
                print('The weighted recall for each label: '+str(recall))
                overall_recall = overall_avg(recall,'recall') #average recall
                f1 = avg_metric(metrics_df,'f1-score')#f-1 scores weighted by the support for all 9 label
                print('The weighted f1-score for each label: '+str(f1))
                overall_f1 = overall_avg(f1,'f1-score')#average f1-score
                metrics_all_nn.append([overall_prec,overall_recall,overall_f1])
    df1 = pd.DataFrame(params)
    df2 = pd.DataFrame(metrics_all_nn)
    df3 = pd.merge(df1, df2, left_index=True,right_index=True)
    df3.columns=['optimizer','dropout','neurons','precision','recall','f1-score']
    return df3
