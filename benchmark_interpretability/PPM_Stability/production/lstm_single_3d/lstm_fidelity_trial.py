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
import math

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict, Counter
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Flatten, Input, LSTM
from keras.optimizers import Nadam, RMSprop
from keras.layers.normalization import BatchNormalization

from tensorflow.keras.backend import print_tensor
from tensorflow.keras.utils import plot_model
from tensorflow.compat.v1 import disable_v2_behavior#, ConfigProto, Session
from tensorflow.compat.v1.keras.backend import get_session
disable_v2_behavior()

import lime
import lime.lime_tabular
from lime import submodular_pick;

import shap

import warnings
warnings.filterwarnings('ignore')

#configure thread pool
#NUM_PARALLEL_EXEC_UNITS = 2
#config = ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
#                        inter_op_parallelism_threads=2, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
#session = Session(config=config)

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

def generate_distributions(explainer, features, test_x, bin_min = -1, bin_max = 1, bin_width = 0.05):
    
        shap_values = explainer.shap_values(test_x, check_additivity = False)
        if type(explainer) == shap.explainers.tree.TreeExplainer:
            shap_val_feat = np.transpose(shap_values)
            feats = np.transpose(test_x)
        elif type(explainer) == shap.explainers.deep.DeepExplainer:
            copy_val = []
            for each in shap_values[0]:
                copy_val.append(list(each.flatten()))
            copy_val = np.array(copy_val)
            shap_val_feat = np.transpose(copy_val)
            
            copy_feat = []
            for each in test_x:
                copy_feat.append(list(each.flatten()))
            copy_feat = np.array(copy_feat)
            feats = np.transpose(copy_feat)
            
        
        features = features.flatten()

        shap_distribs = []

        #For each feature
        for i in range(len(features)):
            print (i+1, "of", len(features), "features")
            shap_vals = shap_val_feat[i]

            #create bins based on shap value ranges
            bins = np.arange(bin_min, bin_max, bin_width)

            feat_vals = []
            for sbin in range(len(bins)):
                nl = []
                feat_vals.append(nl)

            #place relevant feature values into each bin
            for j in range(len(shap_vals)):
                val = shap_vals[j]
                b = 0
                cur_bin = bins[b]
                idx = b

                while val > cur_bin and b < len(bins)-1:
                    idx = b
                    b+=1
                    cur_bin = bins[b]
                feat_vals[idx].append(feats[i][j])

            #Find min and max values for each shap value bin
            mins = []
            maxes = []
            
            for each in feat_vals:
                if each != []:
                    mins.append(min(each))
                    maxes.append(max(each))
             #       width.append("Bin "+str(n))
             #       n+=1
            #plt.bar(width, maxes, bottom = mins)
            #plt.show()

            #Create dictionary with list of bins and max and min feature values for each bin
            feat_name = features[i]

            feat_dict = {'Feature Name': feat_name}
            for each in feat_vals:
                if each != []:
                    mins.append(min(each))
                    maxes.append(max(each))
                else:
                    mins.append(None)
                    maxes.append(None)

            feat_dict['bins'] = bins
            feat_dict['mins'] = mins
            feat_dict['maxes'] = maxes

            shap_distribs.append(feat_dict)
        
        return shap_distribs

dataset_ref = "production"
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
#max_feat = 10

dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "bpic2017" : ["bpic2017_accepted"],
    "bpic2012" : ["bpic2012_accepted"],
    #"insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1"],# "sepsis_cases_2", "sepsis_cases_4"]
    "production" : ["production"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

#Try SHAP
print("----------------------------------------------SHAP----------------------------------------------")

if generate_model_shap:
    for dataset_name in datasets:

        dataset_manager = DatasetManager(dataset_name)

        for ii in range(n_iter):
            if cls_method == "lstm":
                num_buckets = range(1)
            else:
                num_buckets = range(len([name for name in os.listdir(os.path.join(PATH,'%s/%s_%s/models'% (dataset_ref, cls_method, method_name)))]))

            all_shap_changes = []
            all_lens = []
            all_probas = []
            all_case_ids = []

            pos_shap_changes = []
            pos_probas = []
            pos_nr_events = []
            pos_case_ids = []

            neg_shap_changes = []
            neg_probas = []
            neg_nr_events = []
            neg_case_ids = []

            for bucket in list(num_buckets):
                bucketID = "all"
                print ('Bucket', bucketID)

                #import everything needed to sort and predict
                if cls_method == "lstm":
                    print("get everything to create model")
                    params_path = os.path.join(PATH, "%s/%s_%s/cls/params.pickle" % (dataset_ref, cls_method, method_name))
                    with open(params_path, 'rb') as f:
                        args = pickle.load(f)

                    max_len = args['max_len']
                    data_dim = args['data_dim']
                    print("Parameters loaded")

                    #create model
                    print("defining input layer")
                    main_input = Input(shape=(max_len, data_dim), name='main_input')
                    
                    print("adding lstm layers")
                    if args["lstm_layers"]["layers"] == "one":
                        l2_3 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim), implementation=2, 
                                    kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b2_3 = BatchNormalization()(l2_3)

                    if args["lstm_layers"]["layers"] == "two":
                        l1 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim), implementation=2, 
                                kernel_initializer='glorot_uniform', return_sequences=True, 
                                recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b1 = BatchNormalization()(l1)
                        l2_3 = LSTM(args["lstm_layers"]["lstm2_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args["lstm_layers"]["lstm2_dropouts"], stateful = False)(b1)
                        b2_3 = BatchNormalization()(l2_3)
                        
                    if args["lstm_layers"]["layers"] == "three":
                        l1 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim),implementation=2, 
                                kernel_initializer='glorot_uniform', return_sequences=True, 
                                recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b1 = BatchNormalization()(l1)
                        l2 = LSTM(args["lstm_layers"]["lstm2_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, 
                                    recurrent_dropout=args["lstm_layers"]["lstm2_dropouts"], stateful = False)(b1)
                        b2 = BatchNormalization()(l2)
                        l2_3 = LSTM(args["lstm_layers"]["lstm3_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args["lstm_layers"]["lstm3_dropouts"], stateful = False)(b2)
                        b2_3 = BatchNormalization()(l2_3)
                    
                    print("adding dense layers")
                    if args['dense_layers']['layers'] == "two":
                        d1 = Dense(args['dense_layers']['dense2_nodes'], activation = "relu")(b2_3)
                        outcome_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', name='outcome_output')(d1)

                    else:
                        outcome_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', name='outcome_output')(b2_3)
                    
                    print("putting together layers")
                    cls = Model(inputs=[main_input], outputs=[outcome_output])
                    
                    print("choosing optimiser")
                    if args['optimizer'] == "adam":
                        opt = Nadam(lr=args['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
                    elif args['optimizer'] == "rmsprop":
                        opt = RMSprop(lr=args['learning_rate'], rho=0.9, epsilon=1e-08, decay=0.0)
                        
                    print("adding weights to model")
                    checkpoint_path = os.path.join(PATH, "%s/%s_%s/cls/checkpoint.cpt" % (dataset_ref, cls_method, method_name))
                    weights = cls.load_weights(checkpoint_path)
                    #print(weights.assert_consumed())
                     
                    print("compiling model")
                    cls.compile(loss='binary_crossentropy', optimizer=opt)
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
                X_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_prefixes.pickle" % (dataset_ref, cls_method, method_name, bucketID))
                Y_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_labels.pickle" % (dataset_ref, cls_method, method_name, bucketID))

                with open (X_train_path, 'rb') as f:
                    dt_train_bucket = pickle.load(f)
                with open (Y_train_path, 'rb') as f:
                    train_y = pickle.load(f)
                with open (X_test_path, 'rb') as f:
                    dt_test_bucket = pickle.load(f)
                with open (Y_test_path, 'rb') as f:
                    test_y = pickle.load(f)

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
                
                #break

                if cls_method == "xgboost":
                    tree_explainer = shap.TreeExplainer(cls)
                    test_x = feature_combiner.fit_transform(dt_test_bucket)
                    feat_list = feature_combiner.get_feature_names()
                elif cls_method == "lstm":
                    if len(dt_train_bucket) >10000:
                        training_sample = shap.sample(dt_train_bucket, 10000)
                    else:
                        training_sample = dt_train_bucket
                    deep_explainer = shap.DeepExplainer(cls, training_sample)
                    test_x = dt_test_bucket
                    feat_list_path = os.path.join(PATH, "%s/%s_%s/cls/feature_names.pickle" % (dataset_ref, cls_method, method_name))
                    with open(feat_list_path, 'rb') as f:
                        file = f.read()
                        feat_list = np.array(pickle.loads(file))
                #break
                type_list = ['True Negatives', 'True Positives', 'False Negatives', 'False Positives']
                max_feat = round(len(feat_list.flatten())*0.1)
                #print(max_feat)
                
                print("Generating distributions for bucket")
                start_time = time.time()
                if cls_method == "lstm":
                    distribs = generate_distributions(deep_explainer, feat_list, test_x)
                if cls_method == "xgboost":
                    distribs = generate_distributions(tree_explainer, feat_list, test_x)
                dist_elapsed = time.time()-start_time
                print("Time taken to generate distribution:", dist_elapsed)
                
                start_time = time.time()
                for i_type in range(len(sample_instances[:1])):
                    changes = []
                    probas = []
                    nr_events = []
                    case_ids = []

                    for n in range(len(sample_instances[i_type][:1])):
                        print("Category %s of %s. Instance %s of %s" %(i_type+1, len(sample_instances), n+1, len(sample_instances[i_type])))
                        instance = sample_instances[i_type][n]

                        ind = instance['predicted']
                        case_ids.append(instance['caseID'])
                        p1 = instance['proba']
                        probas.append(p1)
                        nr_events.append(instance['nr_events'])
                        input_ = instance['input']

                        if cls_method != "lstm":
                            test_x_group = feature_combiner.fit_transform(input_) 
                        else:
                            test_x_group = np.array([input_])
                        #test_x=np.transpose(test_x_group[0])
                        #print(test_x)
                        #print(p1)

                        print("Creating explanations")
                        if cls_method == "xgboost":
                            exp, rel_exp = create_samples(tree_explainer, exp_iter, test_x_group, feat_list, top = max_feat)
                            features = []
                            shap_vals = []

                            print("Identifying relevant features")
                            for ts in rel_exp:
                                for explanation in ts:
                                    features.extend([feat[0] for feat in explanation])
                                    shap_vals.extend([feat for feat in explanation])

                        elif cls_method == "lstm":
                            exp, rel_exp = create_samples(deep_explainer, exp_iter, test_x_group, feat_list, top = max_feat)
                            features = []
                            shap_vals = []
                        
                            print("Identifying relevant features")
                            for explanation in rel_exp:
                                features.extend([feat[0] for feat in explanation])
                                shap_vals.extend([feat for feat in explanation])

                        counter = Counter(features).most_common(max_feat)

                        feats = [feat[0] for feat in counter]

                        rel_feats = []
                        for feat in feats:
                            vals = [i[1] for i in shap_vals if i[0] == feat]
                            #print(feat, vals)
                            val = np.mean(vals)
                            rel_feats.append((feat, val))

                        intervals = []
                        for item in rel_feats:
                            feat = item [0]
                            val = item[1]

                            print("Creating distribution for feature", rel_feats.index(item)+1, "of", len(rel_feats))

                            if type(feat_list) == list:
                                n = feat_list.index(feat)
                                feat_dict = distribs[n]
                            elif type(feat_list) == np.ndarray:
                                f_ind = int(np.where(feat_list == feat)[0])
                                length = len(feat_list[f_ind])
                                s_ind = int(np.where(feat_list == feat)[1])
                                feat_dict = distribs[f_ind*length+s_ind]
    
                            if feat_dict['Feature Name'] != feat:
                                for each in distribs:
                                    if feat_dict['Feature Name'] == feat:
                                        feat_dict = each

                            bins = feat_dict['bins']
                            mins = feat_dict['mins']
                            maxes = feat_dict['maxes']
                            #print (feat, val, bins, mins, maxes)

                            i = 0
                            while val > bins[i] and i < len(bins)-1:
                                idx = i
                                i+=1
                            #print (i)
                            if mins[i] != None and mins[i] != maxes[i]:
                                min_val = mins[i]
                                max_val = maxes[i]
                            else:
                                j = i
                                while (mins[j] == None or mins[j] == maxes[j]) and j > 0:
                                    min_val = mins[j-1]
                                    max_val = maxes[j-1]
                                    j = j-1

                            interval = max_val - min_val
                            
                            if type(feat_list) == list:
                                index = feat_list.index(feat)
                            elif type(feat_list) == np.ndarray:
                                index = [int(np.where(feat_list==feat)[0]),int(np.where(feat_list==feat)[1])]
                            int_min = max_val
                            int_max = max_val + interval
                            intervals.append((feat, index, int_min, int_max))


                        diffs = []

                        for iteration in range(exp_iter):
                            print("Pertubing - Run", iteration+1)
                            alt_x = np.copy(test_x_group)
                            #print("original:", alt_x)
                            for each in intervals:
                                new_val = random.uniform(each[2], each[3])
                                if cls_encoding == "3d":
                                    alt_x[0][each[1][0]][each[1][1]] = new_val
                                else:
                                    alt_x[0][each[1]] = new_val
                            if cls_method != "lstm":
                                p2 = cls.predict_proba(alt_x)[0][ind]
                            else:
                                p2 = cls.predict(alt_x)[0][ind]
                            diff = p1-p2
                            diffs.append(diff)

                        changes.append(np.mean(diffs))
                        shap_elapsed = time.time()-start_time
                        instance['shap_fid_change'] = diffs
                        #print("RMSE for instance:", np.std(diffs))
                        
                        if ind == 0:
                            pos_shap_changes.append(abs(diff))#np.std(diffs))
                            pos_probas.append(p1)
                            pos_nr_events.append(instance['nr_events'])
                            pos_case_ids.append(instance['caseID'])
                        else:
                            neg_shap_changes.append(abs(diff))#np.std(diffs))
                            neg_probas.append(p1)
                            neg_nr_events.append(instance['nr_events'])
                            neg_case_ids.append(instance['caseID'])

#                     fig, ax = plt.subplots()
#                     ax.plot(probas, changes, 'ro', label = "SHAP")
#                     ax.set_xlabel("Prefix Length")
#                     ax.set_ylabel("Change in prediction probability")
#                     #ax.legend(frameon = False, bbox_to_anchor=(1, 1), loc = 'upper left')
#                     plt.yticks(np.arange(0,1.1, 0.1))
#                     plt.title("Prefix length and change in prediction probability - %s (Bucket %s)" %(type_list[i_type], bucketID))
#                     plt.show()

#                     fig2, ax2 = plt.subplots()
#                     ax2.plot(nr_events, changes, 'ro', label = "SHAP")
#                     ax2.set_xlabel("Prediction Probability")
#                     ax2.set_ylabel("Change in prediction probability")
#                     #ax2.legend(frameon = False, bbox_to_anchor=(1, 1), loc = 'upper left')
#                     plt.yticks(np.arange(0,1.1, 0.1))
#                     plt.title("Prediction probability and change in prediction probability - %s (Bucket %s)" %(type_list[i_type], bucketID))
#                     plt.show()

#                     all_shap_changes.extend(changes)
#                     all_lens.extend(nr_events)
#                     all_probas.extend(probas)
#                     all_case_ids.extend(case_ids)

                #Save dictionaries updated with scores
                with open(tn_path, 'wb') as f:
                    pickle.dump(sample_instances[0], f)
                with open(tp_path, 'wb') as f:
                    pickle.dump(sample_instances[1], f)
                with open(fn_path, 'wb') as f:
                    pickle.dump(sample_instances[2], f)
                with open(fp_path, 'wb') as f:
                    pickle.dump(sample_instances[3], f)


#Try LIME
print("----------------------------------------------LIME----------------------------------------------")
start_time = time.time()
if generate_lime:
    for dataset_name in datasets:

        dataset_manager = DatasetManager(dataset_name)

        for ii in range(n_iter):
            if cls_method == "lstm":
                num_buckets = range(0,1)
            else:
                num_buckets = range(len([name for name in os.listdir(os.path.join(PATH,'%s/%s_%s/models'% (dataset_ref, cls_method, method_name)))]))

            all_lime_changes = []
            all_lens = []
            all_probas = []
            all_case_ids = []

            pos_lime_changes = []
            pos_probas = []
            pos_nr_events = []
            pos_case_ids = []

            neg_lime_changes = []
            neg_probas = []
            neg_nr_events = []
            neg_case_ids = []

            for bucket in list(num_buckets):
                bucketID = "all"
                print ('Bucket', bucketID)

                #import everything needed to sort and predict
                if cls_method == "lstm":
                    print("get everything to create model")
                    params_path = os.path.join(PATH, "%s/%s_%s/cls/params.pickle" % (dataset_ref, cls_method, method_name))
                    with open(params_path, 'rb') as f:
                        args = pickle.load(f)

                    max_len = args['max_len']
                    data_dim = args['data_dim']
                    print("Parameters loaded")

                    #create model
                    print("defining input layer")
                    main_input = Input(shape=(max_len, data_dim), name='main_input')
                    
                    print("adding lstm layers")
                    if args["lstm_layers"]["layers"] == "one":
                        l2_3 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim), implementation=2, 
                                    kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b2_3 = BatchNormalization()(l2_3)

                    if args["lstm_layers"]["layers"] == "two":
                        l1 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim), implementation=2, 
                                kernel_initializer='glorot_uniform', return_sequences=True, 
                                recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b1 = BatchNormalization()(l1)
                        l2_3 = LSTM(args["lstm_layers"]["lstm2_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args["lstm_layers"]["lstm2_dropouts"], stateful = False)(b1)
                        b2_3 = BatchNormalization()(l2_3)
                        
                    if args["lstm_layers"]["layers"] == "three":
                        l1 = LSTM(args['lstm1_nodes'], input_shape=(max_len, data_dim),implementation=2, 
                                kernel_initializer='glorot_uniform', return_sequences=True, 
                                recurrent_dropout=args['lstm1_dropouts'], stateful = False)(main_input)
                        b1 = BatchNormalization()(l1)
                        l2 = LSTM(args["lstm_layers"]["lstm2_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, 
                                    recurrent_dropout=args["lstm_layers"]["lstm2_dropouts"], stateful = False)(b1)
                        b2 = BatchNormalization()(l2)
                        l2_3 = LSTM(args["lstm_layers"]["lstm3_nodes"], activation="sigmoid", 
                                    implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, 
                                    recurrent_dropout=args["lstm_layers"]["lstm3_dropouts"], stateful = False)(b2)
                        b2_3 = BatchNormalization()(l2_3)
                    
                    print("adding dense layers")
                    if args['dense_layers']['layers'] == "two":
                        d1 = Dense(args['dense_layers']['dense2_nodes'], activation = "relu")(b2_3)
                        outcome_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', name='outcome_output')(d1)

                    else:
                        outcome_output = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', name='outcome_output')(b2_3)
                    
                    print("putting together layers")
                    cls = Model(inputs=[main_input], outputs=[outcome_output])
                    
                    print("choosing optimiser")
                    if args['optimizer'] == "adam":
                        opt = Nadam(lr=args['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
                    elif args['optimizer'] == "rmsprop":
                        opt = RMSprop(lr=args['learning_rate'], rho=0.9, epsilon=1e-08, decay=0.0)
                        
                    print("adding weights to model")
                    checkpoint_path = os.path.join(PATH, "%s/%s_%s/cls/checkpoint.cpt" % (dataset_ref, cls_method, method_name))
                    weights = cls.load_weights(checkpoint_path)
                    #print(weights.assert_consumed())
                     
                    print("compiling model")
                    cls.compile(loss='binary_crossentropy', optimizer=opt)
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
                    trainingdata = feature_combiner.fit_transform(dt_train_bucket)

                if cls_method == "lstm":
                    feat_list_path = os.path.join(PATH, "%s/%s_%s/cls/feature_names.pickle" % (dataset_ref, cls_method, method_name))
                    with open (feat_list_path, 'rb') as f:
                        file = f.read()
                        orig_list = np.array(pickle.loads(file))
                        feat_list = orig_list[0]
                    comparison_list = []
                    for i in range(max_len):
                        nl = [name+"_t-"+str(i) for name in feat_list]
                        comparison_list.append(nl)
                    max_feat = round(len(np.array(comparison_list).flatten())*0.1)
                else:
                    feat_list = feature_combiner.get_feature_names()
                    max_feat = round(len(feat_list)*0.1)
                
                class_names=['regular','deviant']# regular is 0, deviant is 1, 0 is left, 1 is right
                if cls_method == "lstm":
                    lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(trainingdata,
                                  feature_names =feat_list,
                                  class_names=class_names, discretize_continuous=True)
                else:
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
                                  feature_names =feat_list,
                                  class_names=class_names, discretize_continuous=True)
                
                type_list = ['True Negatives', 'True Positives', 'False Negatives', 'False Positives']

                for i in list(range(len(sample_instances[:1]))):
                    changes = []
                    probas = []
                    nr_events = []
                    case_ids = []

                    for j in list(range(len(sample_instances[i][:1]))):
                        print("Category %s of %s. Instance %s of %s" %(i+1, len(sample_instances), j+1, len(sample_instances[i])))
                        instance = sample_instances[i][j]
                        
                        #start_time = time.time()
                        
                        ind = instance['predicted']
                        case_ids.append(instance['caseID'])
                        p1 = instance['proba']
                        probas.append(p1)
                        #print("proba:", p1)
                        nr_events.append(instance['nr_events'])
                        input_ = instance['input']

                        if cls_method != "lstm":
                            test_x_group= feature_combiner.fit_transform(group) 
                            test_x=np.transpose(test_x_group[0])
                        else:
                            test_x = input_

                        explanations = []
                        for iteration in range(exp_iter):
                            if cls_method == "lstm":
                                lstm = True
                            else:
                                lstm = False
                            lime_exp = generate_lime_explanations(lime_explainer, test_x, cls, input_, max_feat = max_feat, lstm = lstm)
                            explanation = lime_exp.as_list()
                            explanations.extend(explanation)

                        features = []
                        for explanation in explanations:
                            features.append(explanation[0])
                        #print(features)

                        counter = Counter(features)
                        check_dup = []
                        
                        if cls_encoding == "3d":
                            for feat in np.array(comparison_list).flatten():
                                for feature in counter:
                                    if feat in feature:
                                        check_dup.append(feat)
                        else:
                            for feat in np.array(comparison_list).flatten():
                                for feature in counter:
                                    if feat in feature:
                                        check_dup.append(feat)

                        #print(check_dup)

                        dup_counter = Counter(check_dup)
                        duplicated = [feat for feat in dup_counter if dup_counter[feat] > 1]

                        for each in duplicated:
                            dpls = []
                            vals = []
                            for feat in counter.keys():
                                if each in feat:
                                    dpls.append(feat)
                                    vals.append(counter[feat])
                            keepval = vals.index(max(vals))
                            for n in range(len(dpls)):
                                if n != keepval:
                                    del counter[dpls[n]]

                        rel_feat = counter.most_common(max_feat)
                        #print(len(rel_feat))

                        intervals = []

                        for item in rel_feat:
                            print("Creating distribution for feature", rel_feat.index(item))
                            feat = item[0]
                            #print(item)
                            #print(feat)
                            if cls_encoding == "3d":
                                for n in range(len(comparison_list)):
                                    for m in range(len(comparison_list[n])):
                                        if comparison_list[n][m] in feat:
                                            if ("<" or "<=") in feat and (">" or ">=") in feat:
                                                two_sided = True
                                                parts = feat.split(' ')
                                                l_bound = float(parts[0])
                                                u_bound = float(parts[-1])
                                                interval = u_bound - l_bound
                                                new_min = u_bound
                                                new_max = u_bound + interval
                                            else:
                                                two_sided = False
                                                parts = feat.split(' ')
                                                if parts[-2] == "<=" or parts[-2] == "<":
                                                    u_bound = float(parts[-1])
                                                    if u_bound != 0:
                                                        interval = math.ceil(u_bound*1.1)
                                                    else:
                                                        interval = 5
                                                    new_min = u_bound
                                                    new_max = u_bound + interval
                                                elif parts[-2] == ">=" or parts[-2] == ">":
                                                    l_bound = float(parts[-1])
                                                    if l_bound != 0:
                                                        interval = math.ceil(l_bound*1.1)
                                                    else:
                                                        interval = 5
                                                    new_max = l_bound
                                                    new_min = l_bound - interval
                                                else:
                                                    bound = float(parts[-1])
                                                    interval = math.ceil((bound*1.1)/2)
                                                    new_min = bound
                                                    new_max = bound+interval
                                            feature_name = comparison_list[n][m]
                                            index = [n,m]
                                            int_min = new_min
                                            int_max = new_max
                                            #print(feature_name, index, int_min, int_max)
                                            intervals.append((feature_name, index, int_min, int_max))

                            else:
                                for n in range(len(feat_list)):
                                    if feat_list[n] in feat:
                                        if ("<" or "<=") in feat and (">" or ">=") in feat:
                                            two_sided = True
                                            parts = feat.split(' ')
                                            l_bound = float(parts[0])
                                            u_bound = float(parts[-1])
                                            interval = u_bound - l_bound
                                            new_min = u_bound
                                            new_max = u_bound + interval
                                        else:
                                            two_sided = False
                                            parts = feat.split(' ')
                                            if parts[-2] == "<=" or parts[-2] == "<":
                                                u_bound = float(parts[-1])
                                                if u_bound != 0:
                                                    interval = math.ceil(u_bound*1.1)
                                                else:
                                                    interval = 5
                                                new_min = u_bound
                                                new_max = u_bound + interval
                                            elif parts[-2] == ">=" or parts[-2] == ">":
                                                l_bound = float(parts[-1])
                                                if l_bound != 0:
                                                    interval = math.ceil(l_bound*1.1)
                                                else:
                                                    interval = 5
                                                new_max = l_bound
                                                new_min = l_bound - interval
                                            else:
                                                bound = float(parts[-1])
                                                interval = math.ceil((bound*1.1)/2)
                                                new_min = bound
                                                new_max = bound+interval
                                        feature_name = feat_list[n]
                                        index = n
                                        int_min = new_min
                                        int_max = new_max
                                        #print(feature_name, index, int_min, int_max)
                                        intervals.append((feature_name, index, int_min, int_max))

                        diffs = []
                        for iteration in range(exp_iter):
                            print("Pertubing - Run", iteration+1)
                            alt_x = np.copy(test_x_group)
                            #print("original:", alt_x)
                            for each in intervals:
                                new_val = random.uniform(each[2], each[3])
                                if cls_encoding == "3d":
                                    alt_x[0][each[1][0]][each[1][1]] = new_val
                                else:
                                    alt_x[0][each[1]] = new_val
                                    
                            if cls_method != "lstm":
                                p2 = cls.predict_proba(alt_x)[0][ind]
                            else:
                                p2 = cls.predict(alt_x)[0][ind]
                            diff = p1-p2
                            diffs.append(diff)

                        changes.append(np.mean(diffs))
                        lime_elapsed = time.time()-start_time
                        instance['lime_fid_change'] = diffs
                        #print("RMSE for instance:", np.std(diffs))


                        if ind == 0:
                            pos_lime_changes.append(abs(diff))#np.std(diffs))
                            pos_probas.append(p1)
                            pos_nr_events.append(instance['nr_events'])
                            pos_case_ids.append(instance['caseID'])
                        else:
                            neg_lime_changes.append(abs(diff))#np.std(diffs))
                            neg_probas.append(p1)
                            neg_nr_events.append(instance['nr_events'])
                            neg_case_ids.append(instance['caseID'])

#                     fig, ax = plt.subplots()
#                     ax.plot(nr_events, changes, 'bo', label = "LIME")
#                     ax.set_xlabel("Prefix Length")
#                     ax.set_ylabel("Change in prediction probability")
#                     #ax.legend(frameon = False, bbox_to_anchor=(1, 1), loc = 'upper left')
#                     #plt.yticks(np.arange(0,1.1, 0.1))
#                     plt.title("Prefix length and change in prediction probability - %s (Bucket %s)" %(type_list[i], bucketID))
#                     plt.show()

#                     fig2, ax2 = plt.subplots()
#                     ax2.plot(probas, changes, 'bo', label = "LIME")
#                     ax2.set_xlabel("Prediction Probability")
#                     ax2.set_ylabel("Change in prediction probability")
#                     #ax2.legend(frameon = False, bbox_to_anchor=(1, 1), loc = 'upper left')
#                     plt.yticks(np.arange(0,1.1, 0.1))
#                     plt.title("Prediction probability and change in prediction probability - %s (Bucket %s)" %(type_list[i], bucketID))
#                     plt.show()

                    all_lime_changes.extend(changes)
                    all_lens.extend(nr_events)
                    all_probas.extend(probas)
                    all_case_ids.extend(case_ids)

                #Save dictionaries updated with scores
                with open(tn_path, 'wb') as f:
                    pickle.dump(sample_instances[0], f)
                with open(tp_path, 'wb') as f:
                    pickle.dump(sample_instances[1], f)
                with open(fn_path, 'wb') as f:
                    pickle.dump(sample_instances[2], f)
                with open(fp_path, 'wb') as f:
                    pickle.dump(sample_instances[3], f)

print("Time taken to generate distribution:", dist_elapsed)
print("Time taken to create SHAP explanation:", shap_elapsed)
print("Time taken to create LIME explanation:", lime_elapsed)
