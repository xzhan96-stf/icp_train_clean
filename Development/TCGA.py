from scipy.io import loadmat
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import text
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
import os
from sklearn.metrics import precision_recall_curve, auc, f1_score
from scipy.io import loadmat,savemat

def AUPRC(y_test,probability):
    precision, recall, thresholds = precision_recall_curve(y_test, probability)
    auprc = auc(recall, precision)
    return round(auprc,5)

class CPSC():
    '''
    The CPSC Class
    '''

    def __init__(self, delta, epsilon, t):
        self.delta = delta  # hyperparameter :threshold for shrinking contrast
        self.epsilon = epsilon  # hyperparameter : threshold for label prediction with p_value in CP (used for reliability-based unlabeled data augmentation)
        self.t = t  # data distillation - to make the softmax more soft

    pass

    def fit(self, x_proper_train, y_proper_train, x_cal_train, y_cal_train):
        '''
        This function fit the CPSC model with the data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        '''

        self.num_of_classes = None
        self.class_count = []

        self.dk = None
        self.dk_hat = None
        self.mean_x_k = []
        self.mean_x_overall = []
        self.mean_x_k_hat = []
        self.mean_squared_deviation = []
        self.squared_deviation_overall = []
        self.A_train = []
        self.A_test = []


        y_proper_train = y_proper_train.reshape(-1, )
        y_cal_train = y_cal_train.reshape(-1, )

        classes, self.class_count = np.unique(y_proper_train, return_counts=True)
        self.num_of_classes = len(classes)
        self.mean_x_k, self.mean_x_overall = self.Calculate_raw_centroids(x_proper_train, y_proper_train)
        self.mean_x_k_hat, self.mean_squared_deviation = self.Calculate_nomalized_contrast(x_proper_train,
                                                                                           y_proper_train,
                                                                                           self.mean_x_k,
                                                                                           self.mean_x_overall,
                                                                                           self.delta)
        self.A_train = self.Calculate_A_train_1(x_cal_train, y_cal_train, self.mean_x_k_hat,
                                                self.mean_squared_deviation, self.t)


    def CPSC_clean(self, x_new, y_train, outlier_threshold = 0.1, wrongly_labeled_threshold = -0.2):
        '''
        This function cleans the training data by: 1) removing outliers; 2) flipping likely wrongly labeled data
        params:
        outlier_threshold: the threshold on the maximum p-value for all possible labels to be deemed as an outlier
        wrongly_labeled_threshold: the threshold on the (the nominal label's p-value - the largest p-value other than the nominal label's) to be deemed as wrong label
        '''
        self.wrongly_labeled_idx = [] #Recording the indices of potentially wrongly labeld data
        self.flipped_label = [] #How to flip the labels of the wrongly labeled data
        self.outlier_idx = [] #Recording the indices of potential outliers
        self.A_test, self.c_k, self.sigma_k_matrix = self.Calculate_A_test_1(x_new, self.mean_x_k_hat,
                                                                             self.mean_squared_deviation, self.t)
        self.p_value = self.Calculate_pvalue(self.A_train, self.A_test)
        y_train_clean = np.array(list(y_train))
        for i in range(self.p_value.shape[0]): #Check for each training datum
            p_value_array = self.p_value[i,:] #Select the p-values for all possible labels for the i-th test sample
            if np.max(p_value_array) <= outlier_threshold: #None of the possible labels conform well to the training data
                self.outlier_idx.append(i) #Recording the indices of potential outliers
            else:
                p_nominal_label = p_value_array[y_train[i]] #p-value of the nominal label
                max_p = np.max(np.delete(p_value_array, y_train[i])) #largest p-value among labels other than the nominal label
                if p_nominal_label - max_p <= wrongly_labeled_threshold: #There is a significantly more likely label! We need to flip the original label.
                    self.wrongly_labeled_idx.append(i)
                    flipped_label = np.argmax(p_value_array) #The potentially more correct label!
                    self.flipped_label.append(flipped_label)
                    y_train_clean[i] = flipped_label #flip the label of the potentially wrongly labeled data
        y_train_clean = np.delete(y_train_clean, self.outlier_idx,axis=0) #remove the potential outliers

        return y_train_clean, self.wrongly_labeled_idx, self.flipped_label, self.outlier_idx
        pass

    # def predict_SC(self, x_new):
    #
    #     self.SC_prediction = []  # SC prediction
    #     self.SC_prob = []  # the probability of SC output
    #     self.A_test, self.c_k, self.sigma_k_matrix = self.Calculate_A_test_1(x_new, self.mean_x_k_hat,
    #                                                                          self.mean_squared_deviation, self.t)
    #     self.SC_prediction = np.argmax(self.sigma_k_matrix, axis=1)
    #     assert self.c_k.shape[1] == 2
    #     self.SC_prob = self.c_k
    #
    #     return self.SC_prediction,self.SC_prob

    def Calculate_raw_centroids(self, X, Y):
        '''
        This function calculate the original centroids of the training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        '''

        self.mean_x_k = np.zeros((self.num_of_classes, X.shape[1]))  # x_k_bar matrix(K*D)
        for k in range(self.num_of_classes):
            self.mean_x_k[k] = np.mean(X[Y == k, :], axis=0)  # (1*D)

        self.mean_x_overall = np.mean(X, axis=0).reshape(1,-1)  # mean of features of all samples(1*D)

        return self.mean_x_k, self.mean_x_overall

    def Calculate_nomalized_contrast(self, X, Y, mean_x_k, mean_x_overall, delta):
        '''
        This function calculate the nomalized contrast of the training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        :param mean_x_k: the mean of 'class k' data, with shape (C,D), np.array
        :param mean_x_overall: the mean of all samples, with shape(1,D),np.array
        '''


        self.squared_deviation_overall = np.zeros((1,X.shape[1]))  # s^2 (1*D)

        for k in range(self.num_of_classes):
            squared_deviation_k = np.sum(np.multiply((X[Y == k, :] - mean_x_k[k]), (X[Y == k, :] - mean_x_k[k])), axis=0).reshape(1,-1)
            self.squared_deviation_overall += squared_deviation_k

        self.mean_squared_deviation = (1 / (
                X.shape[0] - self.num_of_classes)) * self.squared_deviation_overall  # Sj^2 (1*D)
        assert self.mean_squared_deviation.shape == (1,X.shape[1])

        self.standard_deviation = np.sqrt(self.mean_squared_deviation)  # Sj (1*D)

        self.dk = (mean_x_k - mean_x_overall) / (self.standard_deviation + 1e-7)  # Normalized contrast of class k, #2*D
        self.dk = np.nan_to_num(self.dk)

        # Shrunk class centroids

        self.dk_hat = np.multiply(np.sign(self.dk), np.maximum(np.abs(self.dk) - self.delta, 0))  # delta is a hyperparameter
        self.mean_x_k_hat = mean_x_overall + np.multiply(self.standard_deviation, self.dk_hat)  # shrunken contrast
        self.remained_feature_index=[i for i in range(self.dk_hat.shape[1]) if all(np.abs(self.dk_hat[:, i]) > 1e-4)]#the features whose d to centroids are all above zeros (<1e-4) after SC


        return self.mean_x_k_hat, self.mean_squared_deviation

    def Calculate_A_train_1(self, X, Y, mean_x_k_hat, mean_squared_deviation, t):
        '''
        This function calculate the nonconformity measurement with training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''
        A_label = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            label = int(Y[i]) # 0 or 1
            c_k = np.zeros(self.num_of_classes)
            sigma_k = self.Calculate_sigma(X[i], mean_x_k_hat, mean_squared_deviation) # Discriminant score
            c_k = np.exp(sigma_k / t) / (np.sum(np.exp(sigma_k /t)) + 1e-7)  # softmax (1*K) to convert the discriminant score to the predicted probability
            if np.isnan(c_k).any(): #If one label is extremely likely
                c_k= np.zeros(self.num_of_classes)
                c_k[np.argmax(sigma_k)]=1
            A_label[i] = 0.5 - 0.5 * (c_k[label] - np.max(np.delete(c_k, label)))
        return A_label

    def Calculate_sigma(self, x_new, mean_x_k_hat, mean_squared_deviation):  # Input only a SINGLE sample once
        '''
        This function calculate the discriminant score of new samples given by users
        :param x_new: the feature matrix of test set with shape (N, D), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''
        sigma_k = np.zeros(self.num_of_classes)
        pi_k = self.class_count / np.sum(self.class_count)  # prior
        contrast = np.zeros((self.num_of_classes,mean_x_k_hat.shape[1]))
        for i in range(contrast.shape[0]):
           contrast[i] = x_new - mean_x_k_hat[i]
        #contrast = np.repeat(x_new,self.num_of_classes,axis=0) - mean_x_k_hat # Broadcast manually
        sigma_k = np.log(pi_k) - np.sum(0.5 * np.multiply(contrast, contrast) / (mean_squared_deviation + 1e-7), axis=1)
        return sigma_k  # (K,)

    def Calculate_A_test_1(self, x_new, mean_x_k_hat, mean_squared_deviation,t):  # Method 1_
        '''
        This function calculate the nonconformal measurement of new samples given by users. Difference from train: all posible labels need to be tested
        :param x_new: the feature matrix of test set with shape (N, D), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''

        c_k = np.zeros((x_new.shape[0], self.num_of_classes))  # possibility of k
        A_k = np.zeros((x_new.shape[0], self.num_of_classes))  # Noncomformity measurement
        self.sigma_k_matrix = np.empty((x_new.shape[0], self.num_of_classes))  # (N_test,K)

        for i in range(x_new.shape[0]):

            sigma_k = self.Calculate_sigma(x_new[i], mean_x_k_hat, mean_squared_deviation)
            self.sigma_k_matrix[i] = sigma_k

            c_k[i] = np.exp(sigma_k/t) / (np.sum(np.exp(sigma_k/t)) + 1e-7)  # (1*K)
            if np.isnan(c_k[i]).any():
                c_k[i]= np.zeros(self.num_of_classes)
                c_k[i][np.argmax(sigma_k)]=1

            for k in range(self.num_of_classes):
                A_k[i][k] = 0.5 - 0.5 * (c_k[i][k] - np.max(np.delete(c_k[i], k)))

        return A_k, c_k, self.sigma_k_matrix

    def Calculate_pvalue(self, A_train, A_test):
        '''
        This function calculate the p_value of new samples given by users
        :param A_train: the nonconformity measurement matrix of training set, with shape of (N, 1), np.array
        :param A_test: the nonconformity measurement matrix of test data, with shape (N,C), np.array
        '''

        self.p_value = np.zeros((A_test.shape[0], self.num_of_classes))
        for m in range(A_test.shape[0]):
            for n in range(self.num_of_classes):
                cnt = len(A_train[A_train >= A_test[m, n]])
                self.p_value[m, n] = (cnt + 1) / (len(A_train) + 1)
        return self.p_value

if __name__ == '__main__':

    # 1. Define directory, load task and data
    data_dir = 'H:\\My Drive\\Paper\\CP Shapley\\TCGA'
    result_dir = 'H:\\My Drive\\Paper\\CP Shapley\\CPSC Shapley\\Proper Train 0.8 Calibration 0.2\\TCGA Subtype Lasso Nolog outlier 0.1 flip -0.8'
    os.chdir(data_dir)

    ### --- RNA Seq --- ###
    all_dataset_raw = np.load('TCGA_X.npy')
    all_label = np.load('TCGA_Y.npy').astype(int)

    #scaler.fit(all_dataset_raw)
    #processed_dataset = scaler.transform(all_dataset_raw)
    processed_dataset = all_dataset_raw
    del all_dataset_raw

    noises = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    #2. dataset partition
    for noise in noises:
        for repeat in range(0,30):
            #2. Partition proper train (polluted data)/calibration/validation/test sets
            x_train,x_test_val,y_train,y_test_val = train_test_split(processed_dataset, all_label, test_size=0.4, stratify = all_label, random_state=9001+repeat)
            x_unknown_train, x_cal_train, y_unknown_train, y_cal_train = train_test_split(x_train, y_train, test_size=0.2, stratify = y_train, random_state=9001+repeat)  # This part is interesting in the design of experiments

            #3. Label permutation for the proper training data
            if noise != 0:
                number_labels_permuted = int(x_unknown_train.shape[0]*noise)
                idx_permuted = np.random.permutation(np.arange(0,number_labels_permuted)) #permute noise % of data's labels

                true_label = list(y_unknown_train[0:number_labels_permuted])
                permuted_label = list(y_unknown_train[idx_permuted])
                y_unknown_train[0:number_labels_permuted] = np.array(permuted_label) # label permutation

            x_pro_train = x_unknown_train # Polluted Proper Train
            y_pro_train = y_unknown_train

            x_val,x_test,y_val,y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, stratify = y_test_val, random_state=9001+repeat) # True validation and test sets

            #4. Clean the data with CPSC and perform the classification with the raw/cleaned data
            deltas = [0.1,0.2,0.3]
            temperatures = [10,100]
            Cs = [0.05,0.1] # Hyperparameter for the feature selection process

            results_dict = {'Delta': [], 'T': [], 'Model': [], 'Lasso_C': [], 'Raw Acc Val': [], 'Clean Acc Val': [],
                            'Raw Acc Test': [], 'Clean Acc Test': [], 'Raw F1 Val': [], 'Clean F1 Val': [],
                            'Raw F1 Test': [], 'Clean F1 Test': []}

            for delta in deltas:
                for temperature in temperatures:
                    for Lasso_C in Cs:

                        print('Delta: {}, T: {}, C: {}'.format(delta, temperature, Lasso_C))
                        ## 4.1 CPSC training
                        ## 4.1.1 SMOTE to treat data imbalancedness for the CPSC training process
                        # sampling = SMOTE(sampling_strategy=1, random_state=9001)
                        X_pro_train_smote, Y_pro_train_smote = x_pro_train, y_pro_train

                        ## 4.1.2 LASSO for feature extraction for the CPSC training process
                        lassoModel = LogisticRegression(penalty='l1', C=Lasso_C, solver='liblinear')
                        lassoModel.fit(X_pro_train_smote, Y_pro_train_smote)
                        featureSelect = SelectFromModel(lassoModel, prefit=True)
                        x_pro_train_lasso = featureSelect.transform(X_pro_train_smote)
                        x_cal_train_lasso = featureSelect.transform(x_cal_train)

                        ## 4.1.3 Train the CPSC clean model with proper training data and perform the cleaning with the calibration data
                        cpsc = CPSC(delta,0.5,temperature) #delta,epsilon,temperature
                        cpsc.fit(x_proper_train=x_pro_train_lasso, y_proper_train = Y_pro_train_smote, x_cal_train=x_cal_train_lasso, y_cal_train=y_cal_train)
                        y_pro_train_clean_smote, wrongly_labeled_idx, flipped_label, outlier_idx = cpsc.CPSC_clean(x_pro_train_lasso, Y_pro_train_smote, outlier_threshold = 0.1, wrongly_labeled_threshold = -0.8)
                        x_pro_train_clean_smote = np.delete(X_pro_train_smote,outlier_idx,axis=0) #remove the outliers
                        print('Wrongly labeled data indices: ', wrongly_labeled_idx)
                        print('Original labels: ', Y_pro_train_smote[wrongly_labeled_idx])
                        print('Flipped labels: ', flipped_label)
                        print('Outlier indices: ', outlier_idx)

                        ## 5 Classification model training
                        #Initialize the models to be tested
                        lda_base = LinearDiscriminantAnalysis(solver='lsqr')
                        lr_base = LogisticRegression(max_iter=200, random_state=9001)
                        #rf_base = RandomForestClassifier()

                        #Model Performance
                        # models = [lda_base,svm_base,rf_base]
                        # model_names = ['LDA','SVM','RF']
                        models = [lda_base,lr_base]
                        model_names = ['LDA','LR']
                        for idx in range(len(model_names)):
                            print('Model: ',model_names[idx])
                            name = model_names[idx]
                            mdl = models[idx]

                            #5.1 Fit the model with raw training data and evaluate the model on the validation set and test set
                            ##4SMOTE to treat data imbalancedness
                            X_train_smote = np.row_stack([X_pro_train_smote, x_cal_train])
                            Y_train_smote = np.concatenate([Y_pro_train_smote, y_cal_train])

                            ## LASSO for feature extraction
                            lassoModel = LogisticRegression(penalty='l1', C=Lasso_C, solver='liblinear',random_state=9001)
                            lassoModel.fit(X_train_smote, Y_train_smote)
                            featureSelect = SelectFromModel(lassoModel, prefit=True)
                            x_train_lasso = featureSelect.transform(X_train_smote)
                            x_val_lasso = featureSelect.transform(x_val)
                            x_test_lasso = featureSelect.transform(x_test)

                            mdl.fit(x_train_lasso,Y_train_smote)
                            y_pred_val = mdl.predict(x_val_lasso)
                            raw_acc_val = accuracy_score(y_val, y_pred_val)
                            raw_F1_val = f1_score(y_val, y_pred_val, average='macro')
                            print('Raw training set, validation set Acc: ', raw_acc_val)
                            print('Raw training set, validation set F1: ', raw_F1_val)

                            y_pred = mdl.predict(x_test_lasso)
                            raw_acc_test = accuracy_score(y_test, y_pred)
                            raw_F1_test = f1_score(y_test, y_pred, average='macro')
                            print('Raw training set, test set Acc: ', raw_acc_test)
                            print('Raw training set, test set F1: ', raw_F1_test)

                            #5.2Fit the model with clean training data and evaluate the model on the validation set and test set
                            X_train_clean_smote = np.row_stack([x_pro_train_clean_smote, x_cal_train])
                            Y_train_clean_smote = np.concatenate([y_pro_train_clean_smote, y_cal_train])

                            ## LASSO for feature extraction
                            lassoModel = LogisticRegression(penalty='l1', C=Lasso_C, solver='liblinear',random_state=9001)
                            lassoModel.fit(X_train_clean_smote, Y_train_clean_smote)
                            featureSelect = SelectFromModel(lassoModel, prefit=True)
                            x_train_clean_lasso = featureSelect.transform(X_train_clean_smote)
                            x_val_lasso = featureSelect.transform(x_val)
                            x_test_lasso = featureSelect.transform(x_test)

                            mdl.fit(x_train_clean_lasso,Y_train_clean_smote)
                            y_pred_val_clean = mdl.predict(x_val_lasso)
                            clean_acc_val = accuracy_score(y_val, y_pred_val_clean)
                            clean_F1_val = f1_score(y_val, y_pred_val_clean, average='macro')
                            print('Clean training set, validation set Acc: ', clean_acc_val)
                            print('Clean training set, validation set F1: ', clean_F1_val)

                            y_pred_clean = mdl.predict(x_test_lasso)
                            clean_acc_test = accuracy_score(y_test, y_pred_clean)
                            clean_F1_test = f1_score(y_test, y_pred_clean, average='macro')
                            print('Clean training set, test set Acc: ', clean_acc_test)
                            print('Clean training set, test set F1: ', clean_F1_test)

                            results_dict['Delta'].append(delta)
                            results_dict['T'].append(temperature)
                            results_dict['Model'].append(name)
                            results_dict['Lasso_C'].append(Lasso_C)

                            results_dict['Raw Acc Val'].append(raw_acc_val)
                            results_dict['Clean Acc Val'].append(clean_acc_val)
                            results_dict['Raw Acc Test'].append(raw_acc_test)
                            results_dict['Clean Acc Test'].append(clean_acc_test)

                            results_dict['Raw F1 Val'].append(raw_F1_val)
                            results_dict['Clean F1 Val'].append(clean_F1_val)
                            results_dict['Raw F1 Test'].append(raw_F1_test)
                            results_dict['Clean F1 Test'].append(clean_F1_test)

                            os.chdir(result_dir)
                            handle = 'Noise level'+ str(noise) +'_Repeat' + str(repeat)
                            savemat(handle + '_Delta ' + str(delta) + '_T ' + str(temperature) + '_Lasso C ' + str(Lasso_C) + '_Model ' + model_names[idx] + '_cleaning_process.mat', \
                                    {'Y_train': Y_train_smote, 'Y_train_clean': Y_train_clean_smote,\
                                     'Y_unknown_train': Y_pro_train_smote, \
                                     'Wrongly_labeled_idx': wrongly_labeled_idx,\
                                     'Original_labels': Y_pro_train_smote[wrongly_labeled_idx],
                                     'Flipped labels': flipped_label, \
                                     'Outlier_idx': outlier_idx})


            df = pd.DataFrame(results_dict)
            df.to_csv(handle+'_Results.csv')