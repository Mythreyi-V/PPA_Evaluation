#Set up to find files
import sys
import os
PATH = "/home/n9455647/PPM_Evaluation/Stability-Experiments/benchmark_interpretability/PPM_Stability/"
sys.path.append(PATH)

#import required modules
import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory
import stability as st #Nogueira, Sechidis, Brown.

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tensorflow.keras.backend import print_tensor
from keras.models import Sequential, Model, load_model
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.compat.v1.keras.backend import get_session
disable_v2_behavior()

import lime
import lime.lime_tabular
from lime import submodular_pick;

import shap

import warnings
warnings.filterwarnings('ignore')

#Define functionsimport matplotlib.pyplot as plt
def imp_df(column_names, importances):
        df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
        return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, num_feat):
        imp_df.columns = ['feature', 'feature_importance']
        b= sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df.head(num_feat), orient = 'h', palette="Blues_r")

def generate_global_explanations(train_X,train_Y, cls, feature_combiner):
    
    print("The number of testing instances is ",len(train_Y))
    print("The total number of columns is",train_X.shape[1]);
    print("The total accuracy is ",cls.score(train_X,train_Y));
       
    sns.set(rc={'figure.figsize':(10,10), "font.size":18,"axes.titlesize":18,"axes.labelsize":18})
    sns.set
    feat_names = feature_combiner.get_feature_names()
    base_imp = imp_df(feat_names, cls.feature_importances_)
    base_imp.head(15)
    var_imp_plot(base_imp, 'Feature importance using XGBoost', 15)
    return base_imp

from lime import submodular_pick
def generate_lime_explanations(explainer,test_xi, cls,test_y, submod=False, test_all_data=None, max_feat = 10, lstm = False):
    
    #print("Actual value ", test_y)
    if lstm:
        exp = explainer.explain_instance(test_xi, cls.predict, num_features=max_feat, labels=[0,1])
    else:
        exp = explainer.explain_instance(test_xi, 
                                 cls.predict_proba, num_features=max_feat, labels=[0,1])
    
    return exp
        
    if submod==True:
        sp_obj=submodular_pick.SubmodularPick(explainer, test_all_data, cls.predict_proba, 
                                      sample_size=20, num_features=num_features,num_exps_desired=4)
        [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations];

def create_samples(shap_explainer, iterations, row, features, top = None):
    length = len(features)
    
    exp = []
    rel_exp = []
    
    for j in range(iterations):
        
        shap_values = shap_explainer.shap_values(row)
        
        importances = []
        
        if type(shap_explainer) == shap.explainers.kernel.KernelExplainer:
            for i in range(length):
                feat = features[i]
                shap_val = shap_values[0][i]
                abs_val = abs(shap_values[0][i])
                entry = (feat, shap_val, abs_val)
                importances.append(entry)
                
        elif type(shap_explainer) == shap.explainers.tree.TreeExplainer:
            for i in range(length):
                feat = features[i]
                shap_val = shap_values[0][i]
                abs_val = abs(shap_values[0][i])
                entry = (feat, shap_val, abs_val)
                importances.append(entry)
        
        elif type(shap_explainer) == shap.explainers.deep.DeepExplainer:
            for i in range(length):
                if len(features.shape) == 2:
                    for j in range(len(features[i])):
                        feat = features[i][j]
                        shap_val = shap_values[0][0][i][j]
                        abs_val = abs(shap_values[0][0][i][j])
                        entry = (feat, shap_val, abs_val)
                        importances.append(entry)
                else:
                    feat = features[i]
                    shap_val = shap_values[0][0][i]
                    abs_val = abs(shap_values[0][0][i])
                    entry = (feat, shap_val, abs_val)
                    importances.append(entry)
    
        importances.sort(key=lambda tup: tup[2], reverse = True)
        
        exp.append(importances)

        rel_feat = []

        if top != None:
            for i in range(top):
                feat = importances[i]
                if feat[2] > 0:
                    rel_feat.append(feat)

            rel_exp.append(rel_feat)
        else:
            rel_exp = exp
        
    return exp, rel_exp

def dispersal(weights, features):
    feat_len = len(features)
    #print(feat_len)
    weights_by_feat = []
       
    for i in list(range(feat_len)):
        feat_weight = []
        for iteration in weights:
            feat_weight.append(iteration[i])
        weights_by_feat.append(feat_weight)
        
    
    dispersal = []
    dispersal_no_outlier = []
    
    for each in weights_by_feat:
        #print("Feature", weights_by_feat.index(each)+1)
        mean = np.mean(each)
        std_dev = np.std(each)
        var = std_dev**2
        
        if mean == 0:
            dispersal.append(0)
            dispersal_no_outlier.append(0)
        #print(each)
        else:
            #dispersal with outliers
            rel_var = var/abs(mean)
            dispersal.append(rel_var)
            
            #dispersal without outliers - remove anything with a z-score higher
            #than 3 (more than 3 standard deviations away from the mean)
            rem_outlier = []
            z_scores = stats.zscore(each)
            #print(z_scores)
            #print("New list:")
            for i in range(len(z_scores)):
                #print(each[i],":",z_scores[i])
                if -3 < z_scores[i] < 3:
                    rem_outlier.append(each[i])
                #print(rem_outlier)
            if rem_outlier != []:
                new_mean = np.mean(rem_outlier)
                if new_mean == 0:
                    dispersal_no_outlier.append(0)
                else:
                    new_std = np.std(rem_outlier)
                    new_var = new_std**2
                    new_rel_var = new_var/abs(new_mean)
                    dispersal_no_outlier.append(new_rel_var)
            else:
                dispersal_no_outlier.append(rel_var)
    #print(dispersal_no_outlier)
    return dispersal, dispersal_no_outlier

#Set up dataset
dataset_ref = "bpic2012"
params_dir = PATH + "params"
results_dir = "results"
bucket_method = "single"
cls_encoding = "3d"
cls_method = "lstm"

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)

generate_samples = False
generate_lime = True
generate_kernel_shap = False
generate_model_shap = True

sample_size = 2
exp_iter = 10
max_feat = 10
max_prefix = 25

dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "bpic2017" : ["bpic2017_accepted"],
    "bpic2012" : ["bpic2012_accepted"]
    #"insurance": ["insurance_activity", "insurance_followup"],
    #"sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

#Try SHAP
print("----------------------------------------------SHAP----------------------------------------------")
start_time = time.time()

if generate_model_shap:
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        #data = dataset_manager.read_dataset()
        
        for ii in range(n_iter):
            if cls_method == "lstm":
                num_buckets = 1
            else:
                num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s_%s/models'% (dataset_ref, cls_method, method_name)))])
            
            for bucket in range(num_buckets):
                bucketID = "all"
                print ('Bucket', bucketID)

                #import everything needed to sort and predict
                if cls_method == "lstm":
                    cls_path = os.path.join(PATH, "%s/%s_%s/cls/pred_cls.h5" % (dataset_ref, cls_method, method_name))
                    pred_cls = load_model(cls_path)
                else:
                    pipeline_path = os.path.join(PATH, "%s/%s_%s/pipelines/pipeline_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    feat_comb_path = os.path.join(PATH, "%s/%s_%s/bucketers_and_encoders/feature_combiner_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    bucketer_path = os.path.join(PATH, "%s/%s_%s/bucketers_and_encoders/bucketer_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    cls_path = os.path.join(PATH, "%s/%s_%s/models/cls_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))

                    predictor = joblib.load(pipeline_path)
                    cls = joblib.load(cls_path)
                    feature_combiner = joblib.load(feat_comb_path)
                    bucketer = joblib.load(bucketer_path)
                    
                #import data for bucket
                X_train_path = os.path.join(PATH, "%s/%s_%s/train_data/bucket_%s_prefixes.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                Y_train_path = os.path.join(PATH, "%s/%s_%s/train_data/bucket_%s_labels.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                
                with open (X_train_path, 'rb') as f:
                    dt_train_bucket = pickle.load(f)
                with open (Y_train_path, 'rb') as f:
                    train_y = pickle.load(f)
                    
                #import test set
                #X_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_prefixes.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                #Y_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_labels.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                #with open(X_test_path, 'rb') as f:
                #    dt_test_bucket = pickle.load(f)
                #with open(Y_test_path, 'rb') as f:
                #    test_y = pickle.load(f)

                #import previously identified samples
                tn_path = os.path.join(PATH, "%s/%s_%s/samples/true_neg_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                tp_path = os.path.join(PATH, "%s/%s_%s/samples/true_pos_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                fn_path = os.path.join(PATH, "%s/%s_%s/samples/false_neg_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                fp_path = os.path.join(PATH, "%s/%s_%s/samples/false_pos_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))

                sample_instances = []

                with open (tn_path, 'rb') as f:
                    tn_list = pickle.load(f)
                with open (tp_path, 'rb') as f:
                    tp_list = pickle.load(f)
                with open (fn_path, 'rb') as f:
                    fn_list = pickle.load(f)
                with open (fp_path, 'rb') as f:
                    fp_list = pickle.load(f)

                #save results to a list
                sample_instances.append(tn_list)
                sample_instances.append(tp_list)
                sample_instances.append(fn_list)
                sample_instances.append(fp_list)

                #create explainers now that can be passed later
                if cls_method == "xgboost":
                    tree_explainer = shap.TreeExplainer(cls)
                elif cls_method == "lstm":
                    deep_explainer = shap.DeepExplainer(pred_cls, dt_train_bucket)

                #explain the chosen instances and find the stability score
                cat_no = 0
                for category in sample_instances:
                    cat_no += 1
                    instance_no = 0
                    
                    for instance in category[:1]:
                        instance_no += 1    
                        print("Category", cat_no, "of", len(sample_instances), ". Testing", instance_no, "of", len(category), ".")

                        group = instance['input']

                        #print(group.shape,instance['actual'], instance['predicted'])
                        if cls_method != "lstm":
                            test_x_group = feature_combiner.fit_transform(group)
                        else:
                            test_x_group = np.array([group])
                            
                        if cls_method == "lstm":
                            feat_list_path = os.path.join(PATH, "%s/%s_%s/cls/feature_names.pickle" % (dataset_ref, cls_method, method_name))
                            with open(feat_list_path, 'rb') as f:
                                file = f.read()
                                feat_list = np.array(pickle.loads(file))
                        else:
                            feat_list = feature_combiner.get_feature_names()

                        #Get Tree SHAP explanations for instance
                        exp, rel_exp = create_samples(deep_explainer, exp_iter, test_x_group, feat_list, top = max_feat)

                        feat_pres = []
                        feat_weights = []

                        print("Computing feature presence in each iteration")
                        for iteration in rel_exp:
                            #print("Computing feature presence for iteration", rel_exp.index(iteration))
                            
                            if cls_encoding == "3d":
                                #The stability measure functions can only handle two dimensional arrays and lists
                                presence_list = [0]*(feat_list.shape[0]*feat_list.shape[1])
                                length = feat_list.shape[1]
                                for i in range(len(feat_list)):
                                    for j in range(len(feat_list[i])):
                                        each = feat_list[i][j]
                                        for explanation in iteration:
                                            if each in explanation[0]:
                                                list_idx = i*length+j
                                                presence_list[list_idx] = 1
                            else:
                                presence_list = [0]*len(feat_list)
                                list_idx = feat_list.index(each)
                                for explanation in iteration:
                                    if each in explanation[0]:
                                        presence_list[list_idx] = 1

                            feat_pres.append(presence_list)

                        print("Computing feature weights in each iteration")                            
                        for iteration in exp:
                            #print("Compiling feature weights for iteration", exp.index(iteration))
                            
                            if cls_encoding == "3d":
                                #The stability measure functions can only handle two dimensional arrays and lists
                                weights = [0]*(feat_list.shape[0]*feat_list.shape[1])
                                length = feat_list.shape[1]
                                for i in range(len(feat_list)):
                                    for j in range(len(feat_list[i])):
                                        each = feat_list[i][j]
                                        for explanation in iteration:
                                            if each in explanation[0]:
                                                list_idx = i*length+j
                                                weights[list_idx] = explanation[1]
                            else:
                                presence_list = [0]*len(feat_list)
                                list_idx = feat_list.index(each)
                                for explanation in iteration:
                                    if each in explanation[0]:
                                        weights[list_idx] = explanation[1]

                            feat_weights.append(weights)

                        stability = st.getStability(feat_pres)
                        print ("Stability:", round(stability,2))
                        instance['tree_shap_stability'] = stability
                        
                        rel_var, second_var = dispersal(feat_weights, feat_list)
                        avg_dispersal = np.mean(rel_var)
                        print ("Dispersal of feature importance:", round(avg_dispersal, 2))
                        instance['shap_weights_dispersal'] = rel_var
                        adj_dispersal = np.mean(second_var)
                        print ("Dispersal with no outliers:", round(adj_dispersal, 2))
                        instance['adjusted_shap_weights_dispersal'] = second_var
                        
                with open(tn_path, 'wb') as f:
                    pickle.dump(sample_instances[0], f)
                with open(tp_path, 'wb') as f:
                    pickle.dump(sample_instances[1], f)
                with open(fn_path, 'wb') as f:
                    pickle.dump(sample_instances[2], f)
                with open(fp_path, 'wb') as f:
                    pickle.dump(sample_instances[3], f)

elapsed = time.time() - start_time

print("Time taken by SHAP:", elapsed)

#Try LIME
print("----------------------------------------------LIME----------------------------------------------")
start_time = time.time()
if generate_lime:
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        
        for ii in range(n_iter):
            if cls_method == "lstm":
                num_buckets = 1
            else:
                num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s_%s/models'% (dataset_ref, cls_method, method_name)))])
            
            for bucket in range(num_buckets):
                bucketID = "all"
                print ('Bucket', bucketID)

                #import everything needed to sort and predict
                if cls_method == "lstm":
                    cls_path = os.path.join(PATH, "%s/%s_%s/cls/pred_cls.h5" % (dataset_ref, cls_method, method_name))
                    cls = load_model(cls_path)
                else:
                    pipeline_path = os.path.join(PATH, "%s/%s_%s/pipelines/pipeline_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    feat_comb_path = os.path.join(PATH, "%s/%s_%s/bucketers_and_encoders/feature_combiner_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    bucketer_path = os.path.join(PATH, "%s/%s_%s/bucketers_and_encoders/bucketer_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))
                    cls_path = os.path.join(PATH, "%s/%s_%s/models/cls_bucket_%s.joblib" % (dataset_ref, cls_method, method_name, bucketID))

                    predictor = joblib.load(pipeline_path)
                    cls = joblib.load(cls_path)
                    feature_combiner = joblib.load(feat_comb_path)
                    bucketer = joblib.load(bucketer_path)
                
                #import data for bucket
                X_train_path = os.path.join(PATH, "%s/%s_%s/train_data/bucket_%s_prefixes.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                Y_train_path = os.path.join(PATH, "%s/%s_%s/train_data/bucket_%s_labels.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                
                with open (X_train_path, 'rb') as f:
                    dt_train_bucket = pickle.load(f)
                with open (Y_train_path, 'rb') as f:
                    train_y = pickle.load(f)
                #with open (X_test_path, 'rb') as f:
                #    dt_test_bucket = pickle.load(f)
                #with open (Y_test_path, 'rb') as f:
                #    test_y = pickle.load(f)

                #import previously identified samples
                tn_path = os.path.join(PATH, "%s/%s_%s/samples/true_neg_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                tp_path = os.path.join(PATH, "%s/%s_%s/samples/true_pos_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                fn_path = os.path.join(PATH, "%s/%s_%s/samples/false_neg_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                fp_path = os.path.join(PATH, "%s/%s_%s/samples/false_pos_bucket_%s_.pickle" % (dataset_ref, cls_method, method_name, bucketID))

                sample_instances = []

                with open (tn_path, 'rb') as f:
                    tn_list = pickle.load(f)
                with open (tp_path, 'rb') as f:
                    tp_list = pickle.load(f)
                with open (fn_path, 'rb') as f:
                    fn_list = pickle.load(f)
                with open (fp_path, 'rb') as f:
                    fp_list = pickle.load(f)

                #save results to a list
                sample_instances.append(tn_list)
                sample_instances.append(tp_list)
                sample_instances.append(fn_list)
                sample_instances.append(fp_list)

                #get the training data as a matrix
                if cls_method == "lstm":
                    trainingdata = dt_train_bucket
                else:
                    trainingdata = feature_combiner.fit_transform(dt_train_bucket);
                
                #print('Generating local Explanations for', instance['caseID'])
                if cls_method == "lstm":
                    feat_list_path = os.path.join(PATH, "%s/%s_%s/cls/feature_names.pickle" % (dataset_ref, cls_method, method_name))
                    with open (feat_list_path, 'rb') as f:
                        file = f.read()
                        orig_list = np.array(pickle.loads(file))
                        feat_list = orig_list[0]
                else:
                    feat_list = feature_combiner.get_feature_names()
                
                #explain the chosen instances and find the stability score
                cat_no = 0
                for category in sample_instances:
                    cat_no += 1
                    instance_no = 0
                    for instance in category[:1]:
                        instance_no += 1
                        
                        print("Category", cat_no, "of", len(sample_instances), ". Testing", instance_no, "of", len(category), ".")
                        
                        group = instance['input']
                        
                        #create explainer now that can be passed later
                        class_names=['regular','deviant']# regular is 0, deviant is 1, 0 is left, 1 is right
                        if cls_method == "lstm":
                            lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(trainingdata,
                                          feature_names =feat_list,
                                          class_names=class_names, discretize_continuous=True)
                        else:
                            lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
                                          feature_names =feat_list,
                                          class_names=class_names, discretize_continuous=True)

                        #print(group.shape,instance['actual'], instance['predicted'])
                        if cls_method != "lstm":
                            test_x_group= feature_combiner.fit_transform(group) 
                            test_x=np.transpose(test_x_group[0])
                        else:
                            test_x = group
    
                        #Get lime explanations for instance
                        feat_pres = []
                        feat_weights = []

                        for iteration in list(range(exp_iter)):
                            print("Run", iteration)
                            
                            lime_exp = generate_lime_explanations(lime_explainer, test_x, cls, instance['actual'], max_feat = len(feat_list), lstm = True)
                            #print(lime_exp.as_list())

                            if cls_encoding == "3d":
                                #The stability measure functions can only handle two dimensional arrays and lists
                                presence_list = [0]*(orig_list.shape[0]*orig_list.shape[1])
                                weights = [0]*(orig_list.shape[0]*orig_list.shape[1])
                                length = orig_list.shape[1]
                                for i in range(len(feat_list)):
                                    #for j in range(len(feat_list[i])):
                                        each = feat_list[i]#[j]
                                        for explanation in lime_exp.as_list():
                                            if each in explanation[0]:
                                                parts = explanation[0].split(' ')
                                                feat_name = parts[0].split('-')
                                                ts = int(feat_name[-1])
                                                list_idx = ts*length+i
                                                weights[list_idx] = explanation[1]
                                                if lime_exp.as_list().index(explanation) < max_feat:
                                                    presence_list[list_idx] = 1

                            else:
                                presence_list = [0]*len(feat_list)
                                weights = [0]*len(feat_list)

                                for each in feat_list:
                                    list_idx = feat_list.index(each)
                                    #print ("Feature", list_idx)
                                    for explanation in lime_exp.as_list():
                                        if each in explanation[0]:
                                            if lime_exp.as_list().index(explanation) < max_feat:
                                                presence_list[list_idx] = 1
                                            weights[list_idx] = explanation[1]

                            feat_pres.append(presence_list)
                            feat_weights.append(weights)

                        stability = st.getStability(feat_pres)
                        print ("Stability:", round(stability,2))
                        instance['lime_stability'] = stability
                        
                        rel_var, second_var = dispersal(feat_weights, feat_list)
                        avg_dispersal = np.mean(rel_var)
                        print ("Dispersal of feature importance:", round(avg_dispersal, 2))
                        instance['lime_weights_dispersal'] = rel_var
                        adj_dispersal = np.mean(second_var)
                        print ("Dispersal with no outliers:", round(adj_dispersal, 2))
                        instance['adjusted_lime_weights_dispersal'] = second_var
                                        
                #Save dictionaries updated with stability scores
                with open(tn_path, 'wb') as f:
                    pickle.dump(sample_instances[0], f)
                with open(tp_path, 'wb') as f:
                    pickle.dump(sample_instances[1], f)
                with open(fn_path, 'wb') as f:
                    pickle.dump(sample_instances[2], f)
                with open(fp_path, 'wb') as f:
                    pickle.dump(sample_instances[3], f)

elapsed = time.time() - start_time

print("Time taken by LIME:", elapsed)