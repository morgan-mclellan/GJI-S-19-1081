import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.colors import LinearSegmentedColormap

#Set path to input data file
path = 'segment_feature_class_data.txt'

#Set event type (sSSE or lSSE)
event_type = 'lSSE'

# Read data
data=pd.read_csv(path, sep=" ")

# Define dataset used for model training

# Replace ETS NaN (i.e. missing values) with ?
data.fillna('?', inplace=True)
# Get data with known values
data= data[data['Dip'].map(lambda x: x != '?')]
data = data[data['Age'].map(lambda x: x != '?')]
data = data[data['Sed_Thick'].map(lambda x: x != '?')]
data = data[data['Vel'].map(lambda x: x != '?')]
data = data[data['Rough'].map(lambda x: x != '?')]
data_ID = data['Sub_Zone']
data = data.drop('Sub_Zone', axis=1)
data_seg = data['Segment']
data = data.drop('Segment', axis=1)
data_lon = data['Longitude']
data = data.drop('Longitude', axis=1)
data_lat = data['Latitude']
data = data.drop('Latitude', axis=1)

# Select data sets with confirmed Y or N, and U

data_Y = data[data[event_type].map(lambda x: x == 'Y')]
data_N = data[data[event_type].map(lambda x: x == 'N')]
data_U = data[data[event_type].map(lambda x: x == 'U')]

# Remove class for each data point to create dataset of only feature data
feature_data_U = data_U.drop('sSSE', axis=1)
feature_data_U = feature_data_U.drop('lSSE', axis=1)

# Store class label for unknown dataset
y_ = data_U[event_type]

data_all = data[data[event_type].map(lambda x: x =='Y' or x == 'N' or x == 'U')]

# Create subset of data where class is known, this will be used as input
data_input = data[data[event_type].map(lambda x: x == 'Y' or x == 'N')]

# Get feature data from input data st=et
Xdata = data_input

# Create separate variable which only contains the scalar values (drop class from dataset)
Xdata = Xdata.drop('sSSE', axis=1)
Xdata = Xdata.drop('lSSE', axis=1)
y = data_input[event_type]



# Print the total number of inputs as well as the number of inputs for each class 
print('Number of inputs: '+str(len(data_input)))
print('Number of "yes" inputs: '+str(len(data_Y)))
print('Number of "no" inputs: '+str(len(data_N)))

# Now prepare data for ML
from sklearn.preprocessing import StandardScaler, RobustScaler, scale
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVM
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
le = preprocessing.LabelEncoder()

# STEP 1 - Feature Scaling for all data - use robust scaler
# This is necessary to assess the effect of individual features
# for algorithms that use linear discriminators (LR, SVM, LDA)
X_scale = RobustScaler().fit_transform(Xdata)


GNB_mean = []
GNB_std = []
GNB_precision = []
GNB_recall = []
LDA_mean = []
LDA_std = []
LDA_precision = []
LDA_recall = []
LR_mean = []
LR_std = []
LR_precision = []
LR_recall = []
SVM_mean = []
SVM_std = []
SVM_precision = []
SVM_recall = []
KNN_mean = []
KNN_std = []
KNN_precision = []
KNN_recall = []
RF_mean = []
RF_std = []
RF_precision = []
RF_recall = []
GNB_pred_array = []
RF_pred_array = []
LR_pred_array = []
LDA_pred_array = []
SVM_pred_array = []
KNN_pred_array = []
SVM_pred_array = []
RF_mean = []
RF_std = []

# Each test will be run 10 times to produce statistics due to nonunique results associated with creating synthetic dataset

for k in range(10):
    #Create synthetic data
    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(X_scale, y)
    smote_data = pd.DataFrame(X_sm)
    smote_data[event_type] = y_sm
    encoded_labels = le.fit_transform(y_sm)

    seed = 7

    # Prepare models
    models = []
    models.append([GNB(), 'GNB'])
    models.append([LDA(solver='lsqr'), 'LDA'])
    models.append([LR(C=1e6, solver='liblinear', class_weight='balanced'), 'LR'])
    models.append([SVM(gamma='scale'), 'SVM'])
    models.append([KNC(n_neighbors=7), 'KNN'])
    models.append([RF(n_estimators=20, max_depth=7), 'RF'])
    results = []
    names = []
    scoring = 'accuracy'

    ###### CROSS-VALIDATION TEST ######

    # For each ML classifier, split the training/testing sets 10 different times
    # and get the mean and std of the accuracy 
    
    
    for model, name in models:
        ssplit = model_selection.ShuffleSplit(n_splits=50,test_size=0.3,random_state=seed)
        cvss_results = model_selection.cross_val_score(model, X_sm, y_sm, cv=ssplit, scoring=scoring)
        results.append(cvss_results)

        if name == 'GNB':
            GNB_mean.append(float(cvss_results.mean()))
            GNB_std.append((cvss_results.std()))
            
        elif name == 'LDA':
            LDA_mean.append(float(cvss_results.mean()))
            LDA_std.append((cvss_results.std()))
            
        elif name == 'LR':
            LR_mean.append(float(cvss_results.mean()))
            LR_std.append((cvss_results.std()))
            
        elif name == 'SVM':
            SVM_mean.append(float(cvss_results.mean()))
            SVM_std.append((cvss_results.std()))
            
        elif name == 'KNN':
            KNN_mean.append(float(cvss_results.mean()))
            KNN_std.append((cvss_results.std()))
        
    # Due to non uniqueness of RF solution, run each test 10 times to produce statistics for each run
    ps = range(10)
    mean_list = []
    std_list =[]
    for i in ps:
        ssplit = model_selection.ShuffleSplit(n_splits=100,test_size=0.3,random_state=seed)
        cvss_results = model_selection.cross_val_score(RF(n_estimators=20,max_depth=7, random_state=i), X_sm, y_sm, cv=ssplit, scoring=scoring)
        mean_list.append(cvss_results.mean())
        std_list.append(cvss_results.std())
        names.append(name)
    RF_mean_sample = sum(mean_list)/len(mean_list)
    RF_std_sample = sum(std_list)/len(std_list)
    RF_mean.append(RF_mean_sample)
    RF_std.append(RF_std_sample)


    ###### PRECISION AND RECALL ######

    # Precision and Recall 1: Gaussian Naive Bayes
    modelGNB = GNB()
    recall_list = []
    precision_list = []
    rs = range(10)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelGNB.fit(X_train, y_train)
        GNB_out = modelGNB.predict(X_test)
        recall = recall_score(y_test, GNB_out)
        precision = precision_score(y_test, GNB_out)
        recall_list.append(recall)
        precision_list.append(precision)
    GNB_recall.append(float(sum(recall_list)/len(rs)))
    GNB_precision.append(float(sum(precision_list)/len(rs)))

    # Precision and Recall 2: Random Forest
    modelRF = RF(n_estimators=20, max_depth=7)
    recall_list = []
    precision_list = []
    rs = range(10)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelRF.fit(X_train, y_train)
        RF_out = modelRF.predict(X_test)
        recall = recall_score(y_test, RF_out)
        precision = precision_score(y_test, RF_out)
        recall_list.append(recall)
        precision_list.append(precision)
    RF_recall.append(float(sum(recall_list)/len(rs)))
    RF_precision.append(float(sum(precision_list)/len(rs)))

    # Precision and Recall 3: Logistic Regression
    modelLR = LR(C=1e6, solver='liblinear', class_weight='balanced')
    recall_list = []
    precision_list = []
    rs = range(10)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelLR.fit(X_train, y_train)
        LR_out = modelLR.predict(X_test)
        recall = recall_score(y_test, LR_out)
        precision = precision_score(y_test, LR_out)
        recall_list.append(recall)
        precision_list.append(precision)
    LR_recall.append(float(sum(recall_list)/len(rs)))
    LR_precision.append(float(sum(precision_list)/len(rs)))

    # Precision and Recall 4: Support Vector Machine
    modelSVM = SVM(C=1000, kernel='linear', probability=True)
    recall_list = []
    precision_list = []
    rs = range(10)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelSVM.fit(X_train, y_train)
        SVM_out = modelSVM.predict(X_test)
        recall = recall_score(y_test, SVM_out)
        precision = precision_score(y_test, SVM_out)
        recall_list.append(recall)
        precision_list.append(precision)
    SVM_recall.append(float(sum(recall_list)/len(rs)))
    SVM_precision.append(float(sum(precision_list)/len(rs)))

    # Precision and Recall 5: K-Neighbors Classifier
    rs = range(10)
    recall_list = []
    precision_list = []
    modelKNC = KNC(n_neighbors=7)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelKNC.fit(X_train, y_train)
        KNC_out = modelKNC.predict(X_test)
        recall = recall_score(y_test, KNC_out)
        precision = precision_score(y_test, KNC_out)
        recall_list.append(recall)
        precision_list.append(precision)
    KNN_recall.append(float(sum(recall_list)/len(rs)))
    KNN_precision.append(float(sum(precision_list)/len(rs)))

    # Precision and Recall 6: LinearDiscriminant Analysis
    modelLDA = LDA(solver='lsqr')
    recall_list = []
    precision_list = []
    rs = range(10)
    for i in rs:  
        X_train, X_test, y_train, y_test = train_test_split(X_sm, encoded_labels, test_size=0.33, random_state=i)
        modelLDA.fit(X_train, y_train)
        LDA_out = modelLDA.predict(X_test)
        recall = recall_score(y_test, LDA_out)
        precision = precision_score(y_test, LDA_out)
        recall_list.append(recall)
        precision_list.append(precision)
    LDA_recall.append(float(sum(recall_list)/len(rs)))
    LDA_precision.append(float(sum(precision_list)/len(rs)))


    ###### PREDICTIONS ######

    # Fit models to training data
    modelGNB.fit(X_sm, y_sm)
    modelRF.fit(X_sm, y_sm)
    modelLR.fit(X_sm, y_sm)
    modelSVM.fit(X_sm, y_sm)
    modelKNC.fit(X_sm, y_sm)
    modelLDA.fit(X_sm, y_sm)
    
    X_ = feature_data_U
  
    X_unknown_scale = RobustScaler().fit_transform(X_)
    X_unknown_scale.reshape(-1,1)
    
    # Predict Y or N class from features

    GNB_preds = modelGNB.predict_proba(X_unknown_scale)
    GNB_preds = GNB_preds[:,1]
    GNB_pred_array.append(GNB_preds)

    LR_preds = modelLR.predict_proba(X_unknown_scale)
    LR_preds = LR_preds[:,1]
    LR_pred_array.append(LR_preds)

    LDA_preds = modelLDA.predict_proba(X_unknown_scale)
    LDA_preds = LDA_preds[:,1]
    LDA_pred_array.append(LDA_preds)

    SVM_preds = modelSVM.predict_proba(X_unknown_scale)
    SVM_preds = SVM_preds[:,1]
    SVM_pred_array.append(SVM_preds)

    KNN_preds = modelKNC.predict_proba(X_unknown_scale)
    KNN_preds = KNN_preds[:,1]
    KNN_pred_array.append(KNN_preds)

    # Due to non uniqueness of RF solution, predict 10 times to produce statistics for each run

    rs = range(10)
    seg_num_list = [0]*len(X_)
    for i in rs:
        modelRF = RF(n_estimators=20, max_depth=7, random_state=i)
        modelRF.fit(X_sm, y_sm)
        y_predRF = modelRF.predict(X_unknown_scale)
        for j in range(len(X_unknown_scale)):
            if y_predRF[j] == 'Y':
                seg_num_list[j] = seg_num_list[j] + 1
    for i in range(len(seg_num_list)): 
        seg_num_list[i] = seg_num_list[i]/10
    RF_pred_array.append(seg_num_list)

print('GNB mean: '+str(np.mean(GNB_mean)))
print('RF mean: '+str(np.mean(RF_mean)))
print('LR mean: '+str(np.mean(LR_mean)))
print('SVM mean: '+str(np.mean(SVM_mean)))
print('KNN mean: '+str(np.mean(KNN_mean)))
print('LDA mean: '+str(np.mean(LDA_mean)))

print('GNB std: '+str(np.mean(GNB_std)))
print('RF std: '+str(np.mean(RF_std)))
print('LR std: '+str(np.mean(LR_std)))
print('SVM std: '+str(np.mean(SVM_std)))
print('KNN std: '+str(np.mean(KNN_std)))
print('LDA std: '+str(np.mean(LDA_std)))

print('GNB precision: '+str(np.mean(GNB_precision)))
print('RF precision: '+str(np.mean(RF_precision)))
print('LR precision: '+str(np.mean(LR_precision)))
print('SVM precision: '+str(np.mean(SVM_precision)))
print('KNN precision: '+str(np.mean(KNN_precision)))
print('LDA precision: '+str(np.mean(LDA_precision)))

print('GNB recall: '+str(np.mean(GNB_recall)))
print('RF recall: '+str(np.mean(RF_recall)))
print('LR recall: '+str(np.mean(LR_recall)))
print('SVM recall: '+str(np.mean(SVM_recall)))
print('KNN recall: '+str(np.mean(KNN_recall)))
print('LDA recall: '+str(np.mean(LDA_recall)))

# Get average predicted probability of SSE occurrence for each ML model
GNB_pred_array = np.array(GNB_pred_array)
GNB_pred_mean = np.mean(GNB_pred_array, 0)

LR_pred_array = np.array(LR_pred_array)
LR_pred_mean = np.mean(LR_pred_array, 0)

LDA_pred_array = np.array(LDA_pred_array)
LDA_pred_mean = np.mean(LDA_pred_array, 0)

KNN_pred_array = np.array(KNN_pred_array)
KNN_pred_mean = np.mean(KNN_pred_array, 0)

SVM_pred_array = np.array(SVM_pred_array)
SVM_pred_mean = np.mean(SVM_pred_array, 0)

RF_pred_mean = np.mean(RF_pred_array, 0)
preds = pd.DataFrame()

preds[0] = GNB_pred_mean
preds[1] = RF_pred_mean
preds[2] = LR_pred_mean
preds[3] = SVM_pred_mean
preds[4] = KNN_pred_mean

weights = [0.3, 1.0, 0.5, 0.5, 0.3]
preds['weighted_pred'] = (preds*weights).sum(axis=1)/sum(weights)

weighted_pred = pd.Series(preds['weighted_pred'], index=y_.index)
RF_pred_mean = pd.Series(RF_pred_mean, index=y_.index)
SVM_pred_mean = pd.Series(SVM_pred_mean, index=y_.index)
KNN_pred_mean = pd.Series(KNN_pred_mean, index=y_.index)
LDA_pred_mean = pd.Series(LDA_pred_mean, index=y_.index)
LR_pred_mean - pd.Series(LR_pred_mean, index=y_.index)
GNB_pred_mean = pd.Series(GNB_pred_mean, index=y_.index)

preds.to_csv('predictions.csv')

###### EVALUATE PREDICTIVE POWER OF FEATURES USING COEFFICIENTS ######

# We can also check the effect of individual features on the fit
# for the algorithms that use a linear discriminator
#paramsLR = pd.Series(modelLR.coef_[0], index=Xdata.columns)
#paramsSVM = pd.Series(modelSVM.coef_[0], index=Xdata.columns)
#paramsLDA = pd.Series(modelLDA.coef_[0], index=Xdata.columns)

# We can also get the standard deviation of those parameters from
# resampling the feature space and fitting the ML models 1000 times
from sklearn.utils import resample
LR_coef = [modelLR.fit(*resample(X_scale, y)).coef_
    for i in range(1000)]

SVM_coef = [modelSVM.fit(*resample(X_scale, y)).coef_
    for i in range(1000)]

LDA_coef = [modelLDA.fit(*resample(X_scale, y)).coef_
    for i in range(1000)]

errLR = np.std(LR_coef, 0)
paramsLR = np.mean(LR_coef, 0)
errSVM = np.std(SVM_coef, 0)
paramsSVM = np.mean(SVM_coef, 0)
errLDA = np.std(LDA_coef, 0)
paramsLDA = np.mean(LDA_coef, 0)

# Print out the parameters and their spread
print('\nLogistic Regression')
print(pd.DataFrame({'effect': paramsLR[0].round(2),
                    'error': errLR[0].round(2)}))
print('\nSupport Vector Machine')
print(pd.DataFrame({'effect': paramsSVM[0].round(2),
                    'error': errSVM[0].round(2)}))
print('\nLinear Discriminant Analysis')
print(pd.DataFrame({'effect': paramsLDA[0].round(2),
                    'error': errLDA[0].round(2)}))

# Since the features have been normalized (using the Robust scaler),
# these parameters indicate their relative effect on predicting "Y" or "N"
# correctly.
