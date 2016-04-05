# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:28:06 2016

THINKFUL DATA SCIENCE
CAPSTONE PROJECT

Lending Club Good/Bad Loan Prediction
Draft V2 (Include all codes)

See Final Report @ https://github.com/blackgenie13/thinkful_ds_projects/blob/master/.ipynb_checkpoints/00_Capstone_Project_Notebook-checkpoint.ipynb

@author: Michael Lin_2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

##############################################################################################
##################                PART A: DATA ACQUISITION                 ###################
################## 'ZIP_2010-2.csv' contain 3rd party zip-code based data  ###################
##############################################################################################

## Desktop Directory
df1 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/LoanStats3a_securev1.csv', skiprows=1)
df2 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/LoanStats3b_securev1.csv', skiprows=1)
df3 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Thinkful/Data/lending-club-project/ZIP_2010-2.csv')

## Laptop Directory
# df1 = pd.read_csv('C:/Users/black/Desktop/lending-club-project/LoanStats3a_securev1.csv', skiprows=1)
# df2 = pd.read_csv('C:/Users/black/Desktop/lending-club-project/LoanStats3b_securev1.csv', skiprows=1)
# df3 = pd.read_csv('C:/Users/black/Desktop/lending-club-project/ZIP_2010-2.csv')

## Quick Analysis
# df1.info()                        # names of all columns
# df1['loan_status'][39784:39800]   # there's a gap/empty row within the data - need to eliminate it
# df1['loan_status'].unique()       # check out all unique column names

## Getting rid of rows with status = "does not meet the credit policy" - no longer valid now.
## This also resulted closing the gap previously mentioned at idx 39784:39800
df1 = df1[df1.loan_status.str.contains("Does not meet the credit policy.") == False]
df3 = df3.rename(columns={'Zip': 'zip_code'})

## Concatinate df1 (2007-2011 loan data) and df2 (2012-2013 loan data)
df12 = pd.concat([df1, df2])
df12.shape

## Here are some code that would turn strings of time into time variables - however, we decided to 
## exclude all time-based variables as predictors as they do not make sense for prediction purposes.
# df12['issue_d'] = df12[pd.to_datetime(df12['issue_d'])
# df12['issue_year'] = pd.DatetimeIndex(df12['issue_d']).year
# df12['issue_month'] = pd.DatetimeIndex(df12['issue_d']).month

## We originally wanted to analyze all loans, but decided to only predict 3-year loans
## So we no longer need to exclude December-2013 loans as they are very close to maturing.
# df2 = df2[df2.issue_d != 'Dec-2013']


##############################################################################################
##################   PART B: DATA CLEANING, ANALYSIS, AND MORE CLEANING    ###################
##################                                                         ###################
##############################################################################################

#######################################  DATA CLEANING  ######################################

## Only extract useful predictors (for loan data 'df12' only)
predictors = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title', \
              'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status', \
              'pymnt_plan', 'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs', \
              'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', \
              'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', \
              'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', \
              'total_rev_hi_lim']
df12 = df12[predictors].copy()

## Dropping 'title' as the columns holds too many different strings - they are messy and
## likely contributes no values as a predictors.  Also many spelling errors.
df12.drop('title', axis=1, inplace=True)

## Dropping 'application_type' because there is only one value, which is "INDIVIDUAL"
df12.drop('application_type', axis=1, inplace=True)

## Dropping 'earliest_cr_line' because it's a time-based predictors, it doesn't make sense 
## to use it as a predictor because, even if it was significant, it would not apply to 
## future borrowers.
df12.drop('earliest_cr_line', axis=1, inplace=True)

# Strip '%' from 'int_rate' and 'revol_util' and reformat them as float type.
df12.int_rate = pd.Series(df12.int_rate).str.replace('%', '').astype(float)
df12.revol_util = pd.Series(df12.revol_util).str.replace('%', '').astype(float)

## We looked at the first word of all 'emp_title' - and found that similar to 'title', the 
## free text field is too messy to contain any useful information as an effective predictor,
## To use this as a predictor would require way too much effort for the scope of this project.
df12.emp_title = df12.emp_title.str.lower()            # Convert all to lower cap
temp = df12['emp_title'].str[0:].str.split(' ', return_type='frame')  # Extract the first word
df12['emp_title'] = temp[0]                            # Overwrite the field with first word
df12.emp_title.value_counts()                          # Look at all the first word
df12.drop('emp_title', axis=1, inplace=True)             # Dropping 'emp_title' as predictor

## Cleaning up the 'emp_length' in exiting string format and convert it into numeric values
df12.emp_length.value_counts()                           # All 'emp_length' value counts
df12.replace('n/a', np.nan, inplace=True)                # Replace 'n/a' with NaN
df12.emp_length.fillna(value=0, inplace=True)            # Fill all NaN with zeros (0)
df12['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
                            # Extract the number in string and replace the cell w/ # only
df12['emp_length'] = df12['emp_length'].astype(int)      # Convert column to integer
df12.emp_length.value_counts()                           # Double-check value_counts

## Exclude all loans that are 60 months (retaining only 36-month loans), and then drop the
## field as a predictor.
df12 = df12[df12['term']==' 36 months']
df12.term.value_counts()
df12.drop('term', axis=1, inplace=True)

## The two 'fico_range_xxx" variables give primiarly the same range - either in range of 
## 4 or 5; therefore, we'll only need to retain one of the fico score along with the 
## range and will drop one of the two fico scores.  We chose to drop 'fico_range_low'.
df12['fico_range'] = df12['fico_range_high'] - df12['fico_range_low']
df12.fico_range.value_counts()
df12.drop('fico_range_low', axis=1, inplace=True)

## Merging the zip code median income dataset ('df3') with Lending Club's dataset ('df12')
## Note that 'df3' contain two other data sources: median income of each zip code and
## whether the zip code is urban or rural or a mixture of both.  Because Lending Club's
## dataset only displays the first 3 digits of the zip codes (i.e. 123xx), we pre-processed
## the zip-code data in Excel by merging all information in the same format as '123xx'.
df = pd.merge(df12, df3, on='zip_code')
df.zip_code.value_counts()

## Adding a few more calculated predictors here (they're self-explainatory)
df['Median_range'] = df['Max_Median'] - df['Min_Median']
df['Dif_median_from_zip'] = df['annual_inc'] - df['Avg_Median']
df['Dif_mean_from_zip'] = df['annual_inc'] - df['Est_Mean']
df['loan_over_income'] = df['loan_amnt']/df['annual_inc']
df['loan_over_median'] = df['loan_amnt']/df['Avg_Median']

## Categorized the target variable 'target' based on 'loan_status' from the dataset
## Good Loans (target=1) are 'Fully Paid', 'Current', and 'In Grace Period'
## Bad Loans (target=0) are 'Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off'
## And then drop the 'loan_status' varabiel
df.loan_status.value_counts()
df['target'] = np.nan
bad_loan = ["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"]
good_loan = ["Fully Paid", "Current", "In Grace Period"]
df.ix[df.loan_status.isin(bad_loan), 'target'] = 0
df.ix[df.loan_status.isin(good_loan), 'target'] = 1
df.groupby(['loan_status', 'target']).loan_status.count().groupby(level=['loan_status','target']).value_counts()
df.drop('loan_status', axis=1, inplace=True)

## IMPUTATION - taking care of the NaN values
## Overalled, we decided to use median() to fill all the NaN values with the exceptions of
## 'mths_since_last_delinq' and 'mths_since_last_record' - for these two, we think it's more
## appropriate to use zeros (0).  As a result the only remaining predictor with NaN that
## would be imputated with median() is 'revol_util'.
imputed_features = df.median()
imputed_features[['mths_since_last_delinq','mths_since_last_record']] = 0
df = df.fillna(imputed_features)

df.isnull().sum()              # Check to see if any NaN value remains


#############################  DATA ANALYSIS & MORE DATA CLEANING ############################

##'Est_tot_income' and 'Est_household' were created to estimate the mean income of each zip code 
## based on 'Pop' and 'income' so they are 100% correlated with exiting variable such as 'Pop'. 
## Here we drop both 'Est_tot_income' and 'Est_household' but retain 'Pop' (population).
df.drop('Est_tot_income', axis=1, inplace=True)
df.drop('Est_household', axis=1, inplace=True)

## It would seem that 'Avg_Median' and 'Est_Mean' are highly correlated at 0.951.
## As a result, we decided to drop 'Est_Mean'
plt.scatter(df.Avg_Median, df.Est_Mean)
plt.show()
np.corrcoef(df.Avg_Median, df.Est_Mean)
df.drop('Est_Mean', axis=1, inplace=True)

## It would seem that 'Dif_median_from_zip' and 'Dif_mean_from_zip' are highly correlated at 0.994.
## As a result, we decided to drop 'Dif_mean_from_zip'
plt.scatter(df.Dif_median_from_zip, df.Dif_mean_from_zip)
plt.show()
np.corrcoef(df.Dif_median_from_zip, df.Dif_mean_from_zip)
df.drop('Dif_mean_from_zip', axis=1, inplace=True)

## 'Avg_Median' and 'Median_range" don't seem to be strongly correlated (at 0.584)
plt.scatter(df.Avg_Median, df.Median_range)
plt.show()
np.corrcoef(df.Avg_Median, df.Median_range)
## 'loan_over_income' and 'loan_over_median' don't seem to be strongly correlated (at 0.549)
plt.scatter(df.loan_over_income, df.loan_over_median)
plt.show()
np.corrcoef(df.loan_over_income, df.loan_over_median)
## 'Min_Median' and 'Median_range' don't seem to be correlated at all (at -0.077)
plt.scatter(df.Min_Median, df.Median_range)
plt.show()
np.corrcoef(df.Min_Median, df.Median_range)
## 'total_acc" and "open_acc" seem to have some correlation at 0.673, we'll retain both as 
## predictors for now.
plt.scatter(df.total_acc, df.open_acc)
plt.show()
np.corrcoef(df.total_acc, df.open_acc)
## 'fico_range_high' and 'int_rate' seem to be somewhat negatively correlated at -0.670, we'll 
## retain both as predictors for now.
plt.scatter(df.fico_range_high, df.int_rate)
plt.show()
np.corrcoef(df.fico_range_high, df.int_rate)
## 'loan_amnt" and "loan_over_median" seem to have somewhat strong correlation at 0.883, we'll 
## retain both as predictors for now.
plt.scatter(df.loan_amnt, df.loan_over_median)
plt.show()
np.corrcoef(df.loan_amnt, df.loan_over_median)

## 'Max_Median' and 'Median_range' seem to be highly correlated at 0.930.
## As a result, we decided to drop "Max_Median" and retain "Median_range"
plt.scatter(df.Max_Median, df.Median_range)
plt.show()
np.corrcoef(df.Max_Median, df.Median_range)
df.drop('Max_Median', axis=1, inplace=True)

## Note: there may be some potential outliers in the variable 'total_rev_hi_lim'.  
## In addition, the varaible isn't complete and it seems to be strongly correlated with 
## 'revol_bal' at 0.7285; therefore, we'll drop 'total_rev_hi_lim' as well.
plt.scatter(df.revol_bal, df.total_rev_hi_lim)
plt.show()
np.corrcoef(df.revol_bal, df.total_rev_hi_lim)
df.drop('total_rev_hi_lim', axis=1, inplace=True)

## Next we identified 4 EXTREME outliers in the predictor 'revol_bal' that is over 
## 0.9-million.  We decided re-assign these four points with values that is the fifth
## highest in the column - 605,627 (at index 155881) so avoid extreme outliers problems.
# plt.hist(df['revol_bal'])
# plt.show()
# plt.boxplot(df['revol_bal'])
# plt.show()
df['revol_bal'].order(ascending=0).head(10)
top_r_bal = df['revol_bal'][155881]
df['revol_bal'][132714] = top_r_bal
df['revol_bal'][50689] = top_r_bal
df['revol_bal'][51162] = top_r_bal
df['revol_bal'][72921] = top_r_bal
df['revol_bal'].order(ascending=0).head(10)

## The following scatter matrix took a long time to run, and we implemented a better
## visulaization for correlation below this section.
# pd.scatter_matrix(df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc']], alpha=0.05, figsize=(10,10), diagonal='hist')
# plt.show()
# pd.scatter_matrix(df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc','dti','fico_range_high','revol_bal','revol_util','Avg_Median','Min_Median','Pop','Dif_mean_median','loan_over_median']], alpha=0.05, figsize=(10,10), diagonal='hist')
# plt.show()

## Scatter Plot using Heat Map to show correlation of the remaining predictors
## In the interest of space, NOT all predictors are shown here.
sns.set(style="white")
corr = df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc','total_acc',\
           'dti','fico_range_high','revol_bal','revol_util','Avg_Median','Min_Median',\
           'Pop','Dif_mean_median','loan_over_median']].corr()  # Compute the correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)               # Generate a mask for the upper triangle
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))                     # Set up the matplotlib figure
cmap = sns.diverging_palette(220, 10, as_cmap=True)       # Generate a custom diverging colormap
sns.heatmap(corr)                      # Draw the heatmap with the mask and correct aspect ratio
plt.show()


##############################################################################################
##################     PART C: ESTABLISH PERFORMANCE METRIC BASELINE       ###################
##################     Calculate Existing Good/Bad Loan Ratio and ROI      ###################
##############################################################################################

## Establish the overall accruacy of good loan (target=1)
accuracy_all = df['target'].mean()
print('The Overall Accuracy of Good Loans is: {:3.4f}%'.format(accuracy_all*100))

## Establish the good loan accuracy broken out by "Grade" assigned by Lending Club
grade_stats = pd.DataFrame(index=(['A','B','C','D','E','F','G']))
grade_stats['accuracy'] = pd.DataFrame(df.groupby(['grade']).target.mean())
# accuracy_grade.ix['A']               # Use this to select by index (e.g. Grade)
print('The Accuracy of Good Loans for each grade is (in percentage %):\n',grade_stats['accuracy']*100)

## Establish the (average) ROI of the overall loan pool (All)
## ROI is defined as (the interest of the loan) for good loan and zeros for bad laons
df['roi'] = df.int_rate * df.target
roi_all = df['roi'].mean()
print('The Overall Annualized ROI of All Loans is: {:3.4f}%'.format(roi_all))
## Double check result (successed!)
(df['target']==0).count()   # Out[]: 172879
(df['roi']==0).count()      # Out[]: 172879

## Establish the ROI of the loans broken out by "Grade" assigned by Lending Club
grade_stats['roi'] = df.groupby(['grade']).roi.mean()
print('The Annualized ROI of Loans for each grade is (in percentage %):\n',grade_stats['roi']*100)

## Establish the Average int rate (regardless good/bad loans) for each grade
grade_stats['int_rate'] = df.groupby(['grade'])['int_rate'].mean()

## Establish the difference between ROI and Average int rate (regardless good/bad loans) for each grade
grade_stats['roi_dif'] = df.groupby(['grade'])['roi'].mean() - df.groupby(['grade'])['int_rate'].mean()

## Record the number of data points for each grade group
grade_stats['count'] = pd.DataFrame(df.groupby(['grade']).target.count())

## Summerizing
print('Performance Metrics broken out by Grades: \n', grade_stats)
# Performance Metrics broken out by Grades: 
#     accuracy        roi   roi_dif  count
# A  0.946723   7.157969 -0.417239  37446
# B  0.899411  10.568706 -1.203595  66399
# C  0.849609  12.793204 -2.281947  40315
# D  0.805593  14.579616 -3.525741  22813
# E  0.784458  16.171055 -4.462497   4890
# F  0.758958  17.162302 -5.422942    921
# G  0.684211  15.376316 -6.890316     95

## INCORRECT WAY TO CALCULATE ROI - DISCARDED
# df.groupby(['grade'])['int_rate'].mean() * df.groupby(['grade'])['target'].mean()
# df.groupby(['grade'])['int_rate'].mean() * df.groupby(['grade'])['target'].mean() - df.groupby(['grade'])['int_rate'].mean()
## Some other code worth noting
# df[df['grade'].isin(['E','F','G'])].groupby(['grade']).target.count()

#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     Overall accuracy is 88.15%; we're dealing with      ###################
##################     very "unbalanced" dataset, and this is an acc-      ###################
##################     uracy tough to beat.  Therefore, we will need       ###################
##################     to balance the data first. And even with bala-      ###################
##################     nced data, we will likely not be able to beat       ###################
##################     the overall ROI at 11.07%.  Therefore, we will      ###################
##################     need to look into improving accuracy and ROI        ###################
##################     within the Grade subgroups instead as Grade A       ###################
##################     and Grade B have very good accuracy already.        ###################
#--------------------------------------------------------------------------------------------#


##############################################################################################
##################      PART D: PREPARING THE DATASETS FOR MODELING        ###################
##################    Balancing Datasets & Break it into subgroup grades   ###################
##############################################################################################

####################################  BALANCING DATASETS  ####################################

from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,\
                               NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler,\
                               SMOTE, SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

## Turn the dataframes into a numpy array and also create a target array
X = df.drop('target',axis=1).values
y = df.target.values

## t_ratio will be used to control how much we should over-sampled target = 0
## We subtract ratio by 2 because we don't want exactly 50/50.
t_ratio =  np.count_nonzero(y==1) / np.count_nonzero(y==0) - 2
verbose = False

## 'Random over-sampling' over-sampling method
OS = OverSampler(ratio=t_ratio, verbose=verbose)
os_X, os_y = OS.fit_transform(X, y)

## Now we break the dataset futher into sub group based on Grade
## Note that we categorized Grade F and G together due to their small sample sizes
df_A = df[df['grade'] == 'A']
df_B = df[df['grade'] == 'B']
df_C = df[df['grade'] == 'C']
df_D = df[df['grade'] == 'D']
df_E = df[df['grade'] == 'E']
df_FG = df[df['grade'].isin(['F','G'])]

''' THERE ARE OTHER ALGORITHEMS FOR BALANCING THE DATATSET; BUT THEY DO NOT WORK WITH STRINGS.
## 'SMOTE' over-sampling method
smote = SMOTE(ratio=t_ratio, verbose=verbose, kind='regular')
smo_X, smo_target = smote.fit_transform(X, target)

## 'SMOTE bordeline 1'
bsmote1 = SMOTE(ratio=t_ratio, verbose=verbose, kind='borderline1')
bs1_X, bs1_target = bsmote1.fit_transform(X, target)

## 'SMOTE bordeline 2'
bsmote2 = SMOTE(ratio=t_ratio, verbose=verbose, kind='borderline2')
bs2_X, bs2_target = bsmote2.fit_transform(X, target)

## 'SMOTE SVM'
svm_args={'class_weight' : 'auto'}
svmsmote = SMOTE(ratio=t_ratio, verbose=verbose, kind='svm', **svm_args)
svs_X, svs_target = svmsmote.fit_transform(X, target)

## 'SMOTE Tomek links'
STK = SMOTETomek(ratio=t_ratio, verbose=verbose)
stk_X, stk_target = STK.fit_transform(X, target)

## 'SMOTE ENN'
SENN = SMOTEENN(ratio=t_ratio, verbose=verbose)
enn_X, enn_target = SENN.fit_transform(X, target)
-----------------------------------------------------------------------------------------'''

##############################################################################################
##################          PART F: LOGISTIC REGRESSION (NUMERIC)          ###################
##################    This model only tested on the numeric predictors     ###################
##############################################################################################
from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,\
                               NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler,\
                               SMOTE, SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import itertools 
from scipy.stats.stats import pearsonr
from sklearn.metrics import confusion_matrix

##########################  COEFFICIENT SIGNIFICANT IDENTIFICATION  ##########################

## We first checked the coefficients here and determined the significant predictors based on p-values
## NOTE that the statsmodels.api is not as robust as we added more predictors - so we has to limit the
## number of predictors going into the first logistic model.  We excluded "annual_inc" and "fico_range",
## which we went back to test those two and they turned out to be insignificant against other predictors.
num_predictors_test = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', \
                       'Avg_Median', 'Pop', 'delinq_2yrs', 'pub_rec', 'revol_bal', \
                       'emp_length', 'Median_range', 'loan_over_median', 'inq_last_6mths', \
                       'mths_since_last_delinq', 'mths_since_last_record', 'loan_over_income', \
                       'open_acc', 'revol_util', 'total_acc','Dif_median_from_zip', \
                       'RU_Ratio', 'Dif_mean_median', 'Min_Median']
                      
df_num_test = df[num_predictors_test].copy()
df_num_test['Intercept'] = 1.0

## Balancing the dataset so that good loans vs bad loans are more balanced
X_num = df_num_test.values
y = df.target.values

t_ratio =  np.count_nonzero(y==1) / np.count_nonzero(y==0) - 2
verbose = False
OS = OverSampler(ratio=t_ratio, verbose=verbose)
os_X_num, os_y = OS.fit_transform(X_num, y)

## Run the logistic regression on the entire dataset and look up the significance of coefficients
logit = sm.Logit(df['target'], df_num_test)
result = logit.fit()
print (result.summary())
#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     By setting alph equal to 0.05, we can conclude      ###################
##################     that 'pop', 'Median_range', and 'mths_since_la      ###################
##################     st_record' can be eliminated as predictirs gi-      ###################
##################     ven their coefficient p-values are all greater      ###################
##################     than 0.05 at " will likely not be able to beat      ###################
##################     the overall ROI at 0.704, 0.982, and 0.668 re-      ###################
##################     spectively.                                         ###################
#--------------------------------------------------------------------------------------------#

##########################    LOGISTIC REGRESSION MODEL FUNCTION    ##########################

## Logistic Regression Function with Performance Outputs
def logistic_regression (X_train, X_test, y_train, y_test, zero_weight=1):
    """Perform logistic regression using Sklearn package
    Note, must already import LogisticRegression from sklearn.linear_model package
    Arguments:
    X_train -- The predictor-only array dataset for training the model
    X_test  -- The predictor-only array dataset for testing the trained model
    y_train -- The response-only array for training the model
    y_test  -- The response-only array for testing the model results
    zero_weight -- The weight to used for regression's class_weight parameter
                   for favoring predicint y=0; use positive integer only
                   Default value is 1 (i.e. no weight)
    """
    ## Fit the logistic regression model with class_wieght 1X-10X on y=0 (favoring y=0)    
    lr = LogisticRegression(class_weight={0: zero_weight})
    lr.fit(X_train, y_train)
    ## Predict test set target values using weighted model and compared accuracy
    y_predicted = lr.predict(X_test)
    confusion = pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)
    a, b = confusion.shape
    a -= 1
    print(confusion)
    print('The MODELED accuracy score of the test/valdiation set is {:2.3}%'.format(accuracy_score(y_test, y_predicted)*100))
    print('The MODELED accuracy on predicted good loans of test/valid. set is {:2.3f}% with {:2.3f}% reduced coverage'.format(confusion[1][a-1]/confusion[1][a]*100, (1-confusion[1][a]/confusion['All'][a])*100))
    print('The ACTUAL accuracy score of the test/validation set is {:2.3}%'.format(np.count_nonzero(y_test==1)/len(y_test)*100))
    roi_num_test_pred = X_test[:,1] * y_predicted * y_test
    print('The PREDICTED Annualized ROI of test/validation set on predicted good loans is: {:2.3f}%'.format(roi_num_test_pred.mean()))
    roi_num_test = X_test[:,1] * y_test
    print('The ACTUAL Annualized ROI of test/validation set on overall true good loans is: {:2.3f}%'.format(roi_num_test.mean()))
    print('\n')    
    return (y_predicted)

##########################     MODEL 1 - USING BALANCED DATASETS    ##########################

## Based on the conclusion, we select the following numeric coffcients:
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', \
                  'Avg_Median', 'delinq_2yrs', 'pub_rec', 'revol_bal', 'emp_length', \
                  'loan_over_median', 'inq_last_6mths', 'mths_since_last_delinq', \
                  'loan_over_income', 'open_acc', 'revol_util', 'total_acc',\
                  'Dif_median_from_zip', 'RU_Ratio', 'Dif_mean_median', 'Min_Median']

df_num = df[num_predictors].copy()
df_num['Intercept'] = 1.0

## Making predictors array and response array
X_num = df_num.values
y = df.target.values

## Split the Original/Unbalanced Data using train_test_split function:
X_num_train, X_num_test, y_train, y_test = train_test_split(X_num, y, test_size = 0.30, random_state=2016)
X_num.shape
y.shape

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_train==1) / np.count_nonzero(y_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
os_X_num_train, os_y_train = OS.fit_transform(X_num_train, y_train)
os_X_num_test, os_y_test = OS.fit_transform(X_num_test, y_test)

## Call the regression function on the Balanced Datasets (both training and test)
y_pre = logistic_regression (os_X_num_train, os_X_num_test, os_y_train, os_y_test)
# OUT[]: Predicted    0.0    1.0    All
# OUT[]: True                          
# OUT[]: 0.0        19996  18519  38515
# OUT[]: 1.0        13920  31907  45827
# OUT[]: All        33916  50426  84342
# OUT[]: The MODELED accuracy score of the test/valdiation set is 61.5%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 63.275% with 40.212% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 54.3%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 4.074%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 6.833%

## Call the regression function using balanced training set and tested on unbalanced (original) set
y_pre = logistic_regression (os_X_num_train, X_num_test, os_y_train, y_test)
# OUT[]: Predicted    0.0    1.0    All
# OUT[]: True                          
# OUT[]: 0.0         3118   2919   6037
# OUT[]: 1.0        13920  31907  45827
# OUT[]: All        17038  34826  51864
# OUT[]: The MODELED accuracy score of the test/valdiation set is 67.5%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 91.618% with 32.851% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 88.4%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 6.625%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.111%

#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     When using balanced data sets, while we were        ###################
##################     able to improve the model accuracy of balance-      ###################
##################     d test sets, we weren't able to imporve either      ###################
##################     the accuracy or the ROI of the original/balan-      ###################
##################     ced test sets using the resulted model. We de-      ###################
##################     cided to try another model w/out balancing.         ###################
#--------------------------------------------------------------------------------------------#

##########################  MODEL 2 - USING CLASS_WEIGHT PARAMETERS ##########################
df_num = df[num_predictors].copy()
df_num['Intercept'] = 1.0
X_num = df_num.values
y = df.target.values

## Set the class_weight parameter at 3 to favor more on predicting y=0
zero_weight = 3
y_pre = logistic_regression (X_num_train, X_num_test, y_train, y_test, zero_weight)
# OUT[]: Predicted   0.0    1.0    All
# OUT[]: True                         
# OUT[]: 0.0         431   5606   6037
# OUT[]: 1.0        1372  44455  45827
# OUT[]: All        1803  50061  51864
# OUT[]: The MODELED accuracy score of the test/valdiation set is 86.5%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 88.802% with 3.476% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 88.4%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.566%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.111%

## Verifying result using Cross Validation on the entire Data Set: n_fold=5
skf = StratifiedKFold(y, n_folds=5, random_state=2016)

true_cv = []
pred_cv = []
accu_cv = []
for train_index, test_index in skf:
    y_pred = logistic_regression (X_num[train_index], X_num[test_index], y[train_index], y[test_index], zero_weight)
    true_cv.append(y[test_index])
    pred_cv.append(y_pred)
    accu_cv.append(accuracy_score(y[test_index], y_pred))

## Here we plot out the confusion matrix of the Cross-Validation Results
## However, note that the importance of the result is the Overall Increased ROI printed above.    
TrueLabel = list(itertools.chain(*true_cv))
PredictedLabel = list(itertools.chain(*pred_cv))
print ('Correlation between the actual and prediction is:', pearsonr(TrueLabel, PredictedLabel)[0], \
       'with p-value',  ("%2.2f" % pearsonr(TrueLabel, PredictedLabel)[1]))

cm = confusion_matrix(PredictedLabel, TrueLabel)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     Using the numeric predictors selected by the        ###################
##################     balanced dataset, we were able to build a log-      ###################
##################     istic model based on original (unbalanced) da-      ###################
##################     taset using 'class_weight' to favor y=0 in or-      ###################
##################     der to produce results with bettero good laon       ###################
##################     predictions but with lower average ROI.             ###################
#--------------------------------------------------------------------------------------------#

########################## MODEL 3 - PREDICTING ON SUB-GROUP GRADES ##########################

##########################         MODEL 3a - GRADE A LOANS         ##########################
df_A = df[df['grade'] == 'A'].copy()
df_A.drop('grade', axis=1, inplace=True)
dfA_num = df_A[num_predictors].copy()
dfA_num['Intercept'] = 1.0
X_A_num = dfA_num.values
y_A = df_A.target.values

X_A_num_train, X_A_num_test, y_A_train, y_A_test = train_test_split(X_A_num, y_A, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_A_train==1) / np.count_nonzero(y_A_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_A_num_train, osy_A_train = OS.fit_transform(X_A_num_train, y_A_train)
osX_A_num_test, osy_A_test = OS.fit_transform(X_A_num_test, y_A_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_A_num_train, X_A_num_test, y_A_train, y_A_test, zero_weight=3)
# OUT[]: Predicted    1.0    All
# OUT[]: True                   
# OUT[]: 0.0          589    589
# OUT[]: 1.0        10645  10645
# OUT[]: All        11234  11234
# OUT[]: The MODELED accuracy score of the test/valdiation set is 94.8%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 94.757% with 0.000% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 94.8%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 7.162%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 7.162%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_A_num_train, osX_A_num_test, osy_A_train, osy_A_test)
# OUT[]: Predicted   0.0    1.0    All
# OUT[]: True                         
# OUT[]: 0.0        5770   4032   9802
# OUT[]: 1.0        4219   6426  10645
# OUT[]: All        9989  10458  20447
# OUT[]: The MODELED accuracy score of the test/valdiation set is 59.6%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 61.446% with 48.853% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 52.1%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 2.286%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 3.935%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_A_num_train, X_A_num_test, osy_A_train, y_A_test)
# OUT[]: Predicted   0.0   1.0    All
# OUT[]: True                        
# OUT[]: 0.0         345   244    589
# OUT[]: 1.0        4219  6426  10645
# OUT[]: All        4564  6670  11234
# OUT[]: The MODELED accuracy score of the test/valdiation set is 60.3%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 96.342% with 40.627% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 94.8%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 4.161%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 7.162%

##########################         MODEL 3b - GRADE B LOANS         ##########################
df_B = df[df['grade'] == 'B'].copy()
df_B.drop('grade', axis=1, inplace=True)
dfB_num = df_B[num_predictors].copy()
dfB_num['Intercept'] = 1.0
X_B_num = dfB_num.values
y_B = df_B.target.values

X_B_num_train, X_B_num_test, y_B_train, y_B_test = train_test_split(X_B_num, y_B, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_B_train==1) / np.count_nonzero(y_B_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_B_num_train, osy_B_train = OS.fit_transform(X_B_num_train, y_B_train)
osX_B_num_test, osy_B_test = OS.fit_transform(X_B_num_test, y_B_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_B_num_train, X_B_num_test, y_B_train, y_B_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.592%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 10.592%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_B_num_train, osX_B_num_test, osy_B_train, osy_B_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 4.450%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 6.292%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_B_num_train, X_B_num_test, osy_B_train, y_B_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 7.491%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 10.592%

##########################         MODEL 3c - GRADE C LOANS         ##########################
df_C = df[df['grade'] == 'C'].copy()
df_C.drop('grade', axis=1, inplace=True)
dfC_num = df_C[num_predictors].copy()
dfC_num['Intercept'] = 1.0
X_C_num = dfC_num.values
y_C = df_C.target.values

X_C_num_train, X_C_num_test, y_C_train, y_C_test = train_test_split(X_C_num, y_C, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_C_train==1) / np.count_nonzero(y_C_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_C_num_train, osy_C_train = OS.fit_transform(X_C_num_train, y_C_train)
osX_C_num_test, osy_C_test = OS.fit_transform(X_C_num_test, y_C_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_C_num_train, X_C_num_test, y_C_train, y_C_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 12.749%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 12.749%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_C_num_train, osX_C_num_test, osy_C_train, osy_C_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 6.616%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 8.169%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_C_num_train, X_C_num_test, osy_C_train, y_C_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.326%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 12.749%

##########################         MODEL 3d - GRADE D LOANS         ##########################
df_D = df[df['grade'] == 'D'].copy()
df_D.drop('grade', axis=1, inplace=True)
dfD_num = df_D[num_predictors].copy()
dfD_num['Intercept'] = 1.0
X_D_num = dfD_num.values
y_D = df_D.target.values
X_D_num_train, X_D_num_test, y_D_train, y_D_test = train_test_split(X_D_num, y_D, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_D_train==1) / np.count_nonzero(y_D_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_D_num_train, osy_D_train = OS.fit_transform(X_D_num_train, y_D_train)
osX_D_num_test, osy_D_test = OS.fit_transform(X_D_num_test, y_D_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_D_num_train, X_D_num_test, y_D_train, y_D_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 13.683%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.561%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_D_num_train, osX_D_num_test, osy_D_train, osy_D_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 9.456%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 10.248%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_D_num_train, X_D_num_test, osy_D_train, y_D_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 13.435%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.561%

##########################         MODEL 3e - GRADE E LOANS         ##########################
df_E = df[df['grade'] == 'E'].copy()
df_E.drop('grade', axis=1, inplace=True)
dfE_num = df_E[num_predictors].copy()
dfE_num['Intercept'] = 1.0
X_E_num = dfE_num.values
y_E = df_E.target.values
X_E_num_train, X_E_num_test, y_E_train, y_E_test = train_test_split(X_E_num, y_E, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_E_train==1) / np.count_nonzero(y_E_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_E_num_train, osy_E_train = OS.fit_transform(X_E_num_train, y_E_train)
osX_E_num_test, osy_E_test = OS.fit_transform(X_E_num_test, y_E_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_E_num_train, X_E_num_test, y_E_train, y_E_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 12.704%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 15.970%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_E_num_train, osX_E_num_test, osy_E_train, osy_E_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.043%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.501%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_E_num_train, X_E_num_test, osy_E_train, y_E_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 13.945%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 15.970%


##########################         MODEL 3f - GRADE FG LOANS        ##########################
df_FG = df[df['grade'].isin(['F','G'])].copy()
df_FG.drop('grade', axis=1, inplace=True)
dfFG_num = df_FG[num_predictors].copy()
dfFG_num['Intercept'] = 1.0
X_FG_num = dfFG_num.values
y_FG = df_FG.target.values

X_FG_num_train, X_FG_num_test, y_FG_train, y_FG_test = train_test_split(X_FG_num, y_FG, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_FG_train==1) / np.count_nonzero(y_FG_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_FG_num_train, osy_FG_train = OS.fit_transform(X_FG_num_train, y_FG_train)
osX_FG_num_test, osy_FG_test = OS.fit_transform(X_FG_num_test, y_FG_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_FG_num_train, X_FG_num_test, y_FG_train, y_FG_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 9.748%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.779%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_FG_num_train, osX_FG_num_test, osy_FG_train, osy_FG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.809%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 13.055%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_FG_num_train, X_FG_num_test, osy_FG_train, y_FG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 15.178%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.779%

##########################        MODEL 3g - GRADE EFG LOANS        ##########################
df_EFG = df[df['grade'].isin(['E','F','G'])].copy()
df_EFG.drop('grade', axis=1, inplace=True)
dfEFG_num = df_EFG[num_predictors].copy()
dfEFG_num['Intercept'] = 1.0
X_EFG_num = dfEFG_num.values
y_EFG = df_EFG.target.values
X_EFG_num_train, X_EFG_num_test, y_EFG_train, y_EFG_test = train_test_split(X_EFG_num, y_EFG, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_EFG_train==1) / np.count_nonzero(y_EFG_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_EFG_num_train, osy_EFG_train = OS.fit_transform(X_EFG_num_train, y_EFG_train)
osX_EFG_num_test, osy_EFG_test = OS.fit_transform(X_EFG_num_test, y_EFG_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_EFG_num_train, X_EFG_num_test, y_EFG_train, y_EFG_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 12.651%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.234%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_EFG_num_train, osX_EFG_num_test, osy_EFG_train, osy_EFG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.129%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 12.021%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_EFG_num_train, X_EFG_num_test, osy_EFG_train, y_EFG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 14.858%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.234%

#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     Using the numeric predictors and running the        ###################
##################     regression model for each Grade sub-group, we       ###################
##################     are still unable to identify a lift in the ROI      ###################
##################     metrics.  While we can increase good loan pre-      ###################
##################     diction accuracy, average prediction ROI are        ###################
##################     still lower in all subgroup of grades.              ###################
#--------------------------------------------------------------------------------------------#

##############################################################################################
##################            PART G: LOGISTIC REGRESSION (ALL)            ###################
################## This model tested on numeric and categorical predictors ###################
##############################################################################################

## CONTINUE FROM NUMERIC-ONLY PREDICTOR MODEL - WE ADDED THE CATEGROICAL PREDICTORS
## Right off the bat, we decided to elminate 'papymnt_plan' because one of the value only has a sample size of 2
## Right off the bat, we also decided to eliminate 'zip_code' as there are just too many values
## Note that we only retained the numeric predictors that worked in Model 1
## Finally, we later further eliminate 'sub_grade' because its significant is captured in 'grade' already -
## in particular grade 'A'.
logistic_predictors = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', \
       'emp_length', 'home_ownership', 'verification_status','purpose', 'addr_state', \
       'dti', 'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', \
       'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'initial_list_status', \
       'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Cat', 'RU_Ratio']

df_lr = df[logistic_predictors].copy()

##########################         CREATING DUMMY VARIABLES         ##########################

## Prep the dataframe with dummy variables (for the categorical predictors)
## Note that each of the first dummy variable is skipped and not retained in the dataframe.
dummified = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'RU_Cat']
df_lr = df_lr.drop(dummified, axis=1)

dummy_grade = pd.get_dummies(df['grade'], prefix='d_grade')
df_lr = df_lr.join(dummy_grade.ix[:, :'d_grade_F'])
## As mentioned above, 'subgrade' isn't significant because there is no significant difference within each grade
# dummy_subgrade = pd.get_dummies(df['sub_grade'], prefix='d_sub_grade')
# df_lr = df_lr.join(dummy_subgrade.ix[:, 1:])
dummy_home_ownership = pd.get_dummies(df['home_ownership'], prefix='d_home_ownership')
df_lr = df_lr.join(dummy_home_ownership.ix[:, 1:])
dummy_verification_status = pd.get_dummies(df['verification_status'], prefix='d_verification_status')
df_lr = df_lr.join(dummy_verification_status.ix[:, 1:])
dummy_purpose = pd.get_dummies(df['purpose'], prefix='d_purpose')
df_lr = df_lr.join(dummy_purpose.ix[:, 1:])
dummy_addr_state = pd.get_dummies(df['addr_state'], prefix='d_addr_state')
df_lr = df_lr.join(dummy_addr_state.ix[:, 1:])
dummy_initial_list_status = pd.get_dummies(df['initial_list_status'], prefix='d_initial_list_status')
df_lr = df_lr.join(dummy_initial_list_status.ix[:, 1:])
dummy_RU_Cat = pd.get_dummies(df['RU_Cat'], prefix='d_RU_Cat')
df_lr = df_lr.join(dummy_RU_Cat.ix[:, 1:])

df_lr['intercept'] = 1.0

##########################    LOOKING AT THE COEFFICIENT P-VALUE    ##########################

## Here we use the statsmodels to check the p-value of each coefficients.  Note that we had to use
## 'method = 'basinhopping' because the default method somehow doesn't work with our model when not using
## 'dummy_subgrade' as predictors... not sure why here.
logit = sm.Logit(df['target'], df_lr)
logit = sm.Logit(df['target'], df_num_test)
result = logit.fit(method='basinhopping')
print (result.summary())

##########################     MODEL 1 - USING ORIGINAL DATASETS    ##########################

## Turning the predictor dataframe into an array.  Note that "target" is already an array
X_lr = df_lr.values
y = df.target.values

## Split the Data using train_test_split function:
X_lr_train, X_lr_test, y_train, y_test = train_test_split(X_lr, y, test_size = 0.30, random_state=0)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_lr_train, X_lr_test, y_train, y_test, zero_weight=3)
# OUT[]: Predicted   0.0    1.0    All
# OUT[]: True                         
# OUT[]: 0.0         377   5828   6205
# OUT[]: 1.0        1045  44614  45659
# OUT[]: All        1422  50442  51864
# OUT[]: The MODELED accuracy score of the test/valdiation set is 86.7%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 88.446% with 2.742% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 88.0%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.671%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.088%

##########################     MODEL 2 - USING BALANCED DATASETS    ##########################

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_train==1) / np.count_nonzero(y_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
os_X_lr_train, os_y_train = OS.fit_transform(X_lr_train, y_train)
os_X_lr_test, os_y_test = OS.fit_transform(X_lr_test, y_test)

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (os_X_lr_train, os_X_lr_test, os_y_train, os_y_test)
# OUT[]: Predicted    0.0    1.0    All
# OUT[]: True                          
# OUT[]: 0.0        20559  19640  40199
# OUT[]: 1.0        14086  31573  45659
# OUT[]: All        34645  51213  85858
# OUT[]: The MODELED accuracy score of the test/valdiation set is 60.7%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 61.650% with 40.352% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 53.2%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 3.929%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 6.698%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (os_X_lr_train, X_lr_test, os_y_train, y_test)
# OUT[]: Predicted    0.0    1.0    All
# OUT[]: True                          
# OUT[]: 0.0         3176   3029   6205
# OUT[]: 1.0        14086  31573  45659
# OUT[]: All        17262  34602  51864
# OUT[]: The MODELED accuracy score of the test/valdiation set is 67.0%
# OUT[]: The MODELED accuracy on predicted good loans of test/valid. set is 91.246% with 33.283% reduced coverage
# OUT[]: The ACTUAL accuracy score of the test/validation set is 88.0%
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 6.504%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.088%

##########################       MODEL 3 - USING ONLY GRADE 'D'     ##########################
df_D = df[df['grade'] == 'D'].copy()
dfD_lr = df_lr[df_lr['d_grade_D'] == 1].copy()
dfD_lr.drop(['d_grade_A','d_grade_B','d_grade_C','d_grade_D','d_grade_E','d_grade_F'] , axis=1, inplace=True)

X_D_lr = dfD_lr.values
y_D = df_D.target.values
X_D_lr_train, X_D_lr_test, y_D_train, y_D_test = train_test_split(X_D_lr, y_D, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_D_train==1) / np.count_nonzero(y_D_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_D_lr_train, osy_D_train = OS.fit_transform(X_D_lr_train, y_D_train)
osX_D_lr_test, osy_D_test = OS.fit_transform(X_D_lr_test, y_D_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_D_lr_train, X_D_lr_test, y_D_train, y_D_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 13.690%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.561%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_D_lr_train, osX_D_lr_test, osy_D_train, osy_D_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 9.490%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 10.248%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_D_lr_train, X_D_lr_test, osy_D_train, y_D_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 13.484%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.561%


##########################   MODEL 4 - USING ONLY GRADE 'E','F','G' ##########################
df_EFG = df[df['grade'].isin(['E','F','G'])].copy()
dfEFG_lr = df_EFG[logistic_predictors].copy()
dfEFG_lr.drop('grade', axis=1, inplace=True)

dummified = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'RU_Cat']
dfEFG_lr = dfEFG_lr.drop(dummified, axis=1)
dummy_home_ownership = pd.get_dummies(df['home_ownership'], prefix='d_home_ownership')
dfEFG_lr = dfEFG_lr.join(dummy_home_ownership.ix[:, 1:])
dummy_verification_status = pd.get_dummies(df['verification_status'], prefix='d_verification_status')
dfEFG_lr = dfEFG_lr.join(dummy_verification_status.ix[:, 1:])
dummy_purpose = pd.get_dummies(df['purpose'], prefix='d_purpose')
dfEFG_lr = dfEFG_lr.join(dummy_purpose.ix[:, 1:])
dummy_addr_state = pd.get_dummies(df['addr_state'], prefix='d_addr_state')
dfEFG_lr = dfEFG_lr.join(dummy_addr_state.ix[:, 1:])
dummy_initial_list_status = pd.get_dummies(df['initial_list_status'], prefix='d_initial_list_status')
dfEFG_lr = dfEFG_lr.join(dummy_initial_list_status.ix[:, 1:])
dummy_RU_Cat = pd.get_dummies(df['RU_Cat'], prefix='d_RU_Cat')
dfEFG_lr = dfEFG_lr.join(dummy_RU_Cat.ix[:, 1:])

dfEFG_lr['Intercept'] = 1.0

X_EFG_lr = dfEFG_lr.values
y_EFG = df_EFG.target.values
X_EFG_lr_train, X_EFG_lr_test, y_EFG_train, y_EFG_test = train_test_split(X_EFG_lr, y_EFG, test_size = 0.30, random_state=2016)

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_EFG_train==1) / np.count_nonzero(y_EFG_train==0) - 2
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_EFG_lr_train, osy_EFG_train = OS.fit_transform(X_EFG_lr_train, y_EFG_train)
osX_EFG_lr_test, osy_EFG_test = OS.fit_transform(X_EFG_lr_test, y_EFG_test)

## Logistic Regression Using Original (Unbalanced) Datasets for Training and Testing with zero_weight at 3
y_pre = logistic_regression (X_EFG_lr_train, X_EFG_lr_test, y_EFG_train, y_EFG_test, zero_weight=3)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 12.291%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.234%

## Logistic Regression Using Balanced Datasets for Training and Testing with no zero_weight
y_pre = logistic_regression (osX_EFG_lr_train, osX_EFG_lr_test, osy_EFG_train, osy_EFG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 10.946%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 12.021%

## Logistic Regression Using Balanced Datasets for Training and Original/Unbalanced for Testing with no zero_weight
y_pre = logistic_regression (osX_EFG_lr_train, X_EFG_lr_test, osy_EFG_train, y_EFG_test)
# OUT[]: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 14.782%
# OUT[]: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.234%

#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     Pretty much the same results as before.             ###################
#--------------------------------------------------------------------------------------------#

##############################################################################################
##################           PART H: RANDOM FORREST DECISION TREE          ###################
################## This model tested on numeric and categorical predictors ###################
##############################################################################################

from sklearn.ensemble import RandomForestClassifier as RFC

var = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'pymnt_plan', 'purpose', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal','revol_util', 
       'total_acc', 'initial_list_status', 'fico_range', 'Avg_Median', 'Min_Median', 
       'Pop', 'Dif_mean_median', 'Median_range','Dif_median_from_zip', 'loan_over_income', 
       'loan_over_median', 'RU_Cat', 'RU_Ratio']

df_rf = df.copy()

df_rf['grade'] = pd.Categorical.from_array(df.grade).codes
df_rf['sub_grade'] = pd.Categorical.from_array(df.sub_grade).codes
df_rf['home_ownership'] = pd.Categorical.from_array(df.home_ownership).codes
df_rf['verification_status'] = pd.Categorical.from_array(df.verification_status).codes
df_rf['addr_state'] = pd.Categorical.from_array(df.addr_state).codes
df_rf['initial_list_status'] = pd.Categorical.from_array(df.initial_list_status).codes
df_rf['RU_Cat'] = pd.Categorical.from_array(df.RU_Cat).codes
df_rf['zip_code'] = pd.Categorical.from_array(df.zip_code).codes
df_rf['pymnt_plan'] = pd.Categorical.from_array(df.pymnt_plan).codes
df_rf['purpose'] = pd.Categorical.from_array(df.purpose).codes

### SPLITTING DATA INTO TRAINING AND TESTING SETS
np.random.seed(2016)
msk = np.random.rand(len(df_rf)) < 0.70
df_rf_train = df_rf[msk]
df_rf_test = df_rf[~msk]
# Alternatively (but cannot reproduce this result)
#  from sklearn.cross_validation import train_test_split
#  df_train, df_test = train_test_split(df, test_size = 0.7)

## Factorized the target, note that sort must be equal True to rain 1 and 0 integrity
y_train, _ = pd.factorize(df_rf_train['target'], sort=True)
y_test, _ = pd.factorize(df_rf_test['target'], sort=True)

## Converting df_rf dataframe into predictor arrays
X_train = df_rf_train.values
X_test = df_rf_test.values

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_train==1) / np.count_nonzero(y_train==0)
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_train, osy_train = OS.fit_transform(X_train, y_train)
osX_test, osy_test = OS.fit_transform(X_test, y_test)

## Converting array back to dataframe.
columns = df_rf.columns
osdf_rf_train = pd.DataFrame(data=osX_train, columns=columns)
osdf_rf_test = pd.DataFrame(data=osX_test, columns=columns)

##########################      RANDOM FORREST MODEL FUNCTION       ##########################

## Logistic Regression Function with Performance Outputs
def random_forrest (df_Xtrain, df_Xtest, df_ytrain, df_ytest, n_jobs=-1, n_est=500, w_start=False, zero_weight=1):
    """Perform Random Forrest Decision using sklearn.ensemble package
        Note, must already import RandomForestClassifier as RFC from sklearn.ensemble package
    Arguments:
    X_train -- The predictor-only array dataset for training the model
    X_test  -- The predictor-only array dataset for testing the trained model
    y_train -- The response-only array for training the model
    y_test  -- The response-only array for testing the model results
    zero_weight -- The weight to used for favoring zero (however, it isn't working for this particular model) 
    """
    # int_rate = df_Xtest['int_rate']
    # df_Xtrain.drop('int_rate', axis=1, inplace=True)
    # df_Xtest.drop('int_rate', axis=1, inplace=True)    
    
    ## Fit the random forrest model with
    forest = RFC(n_jobs=n_jobs, n_estimators=n_est, warm_start=w_start, class_weight={0:zero_weight})    
    forest.fit(df_Xtrain, df_ytrain)

    ## Predict test set target values using weighted model and compared accuracy
    y_predicted = forest.predict(df_Xtest)
    confusion = pd.crosstab(df_ytest, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)
    a, b = confusion.shape
    a -= 1
    print(confusion)
    print('The MODELED accuracy score of the test/valdiation set is {:2.3}%'.format(accuracy_score(df_ytest, y_predicted)*100))
    print('The MODELED accuracy on predicted good loans of test/valid. set is {:2.3f}% with {:2.3f}% reduced coverage'.format(confusion[1][a-1]/confusion[1][a]*100, (1-confusion[1][a]/confusion['All'][a])*100))
    print('The ACTUAL accuracy score of the test/validation set is {:2.3}%'.format(np.count_nonzero(df_ytest==1)/len(df_ytest)*100))
    roi_num_test_pred = df_Xtest['int_rate'] * y_predicted * df_ytest
    print('The PREDICTED Annualized ROI of test/validation set on predicted good loans is: {:2.3f}%'.format(roi_num_test_pred.mean()))
    roi_num_test = df_Xtest['int_rate'] * df_ytest
    print('The ACTUAL Annualized ROI of test/validation set on overall true good loans is: {:2.3f}%'.format(roi_num_test.mean()))
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    features = df_rf[var].columns[:]
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    #plt.savefig('foo.png')
    plt.show()
    print('\n')
    return (y_predicted)

##########################     MODEL 1 - USING BALANACED DATASET    ##########################

y_pred = random_forrest (osdf_rf_train[var], osdf_rf_test[var], osdf_rf_train['target'], osdf_rf_test['target'], \
                         n_jobs=-1, n_est=500, w_start=False, zero_weight=1)
# OUT []: Predicted  0.0    1.0    All
# OUT []: True                        
# OUT []: 0.0        147  39429  39576
# OUT []: 1.0         47  45665  45712
# OUT []: All        194  85094  85288
# OUT []: The MODELED accuracy score of the test/valdiation set is 53.7%
# OUT []: The MODELED accuracy on predicted good loans of test/valid. set is 53.664% with 0.227% reduced coverage
# OUT []: The ACTUAL accuracy score of the test/validation set is 53.6%
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 6.726%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 6.736%

y_pred = random_forrest (osdf_rf_train[var], df_rf_test[var], osdf_rf_train['target'], df_rf_test['target'], \
                         n_jobs=-1, n_est=500, w_start=True, zero_weight=1)
# OUT []: Predicted  0.0    1.0    All
# OUT []: True                        
# OUT []: 0.0         22   6121   6143
# OUT []: 1.0         42  45670  45712
# OUT []: All         64  51791  51855
# OUT []: The MODELED accuracy score of the test/valdiation set is 88.1%
# OUT []: The MODELED accuracy on predicted good loans of test/valid. set is 88.181% with 0.123% reduced coverage
# OUT []: The ACTUAL accuracy score of the test/validation set is 88.2%
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.063%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.079%

##########################  MODEL 2 - USING ORIGINAL BALANCED DATA  ##########################

y_pred = random_forrest (df_rf_train[var], df_rf_test[var], df_rf_train['target'], df_rf_test['target'], \
                         n_jobs=-1, n_est=500, w_start=False, zero_weight=1)
# OUT []: Predicted  0.0    1.0    All
# OUT []: True                        
# OUT []: 0.0          1   6142   6143
# OUT []: 1.0          1  45711  45712
# OUT []: All          2  51853  51855
# OUT []: The MODELED accuracy score of the test/valdiation set is 88.2%
# OUT []: The MODELED accuracy on predicted good loans of test/valid. set is 88.155% with 0.004% reduced coverage
# OUT []: The ACTUAL accuracy score of the test/validation set is 88.2%
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.079%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.079%

##########################       MODEL 3 - USING CLASS WEIGHT       ##########################

for i in range(2,12,2):
    print ('Zero_weight is {}'.format(i))
    y_pred = random_forrest (df_rf_train[var], df_rf_test[var], df_rf_train['target'], df_rf_test['target'], \
                             n_jobs=-1, n_est=500, w_start=False, zero_weight=i)
# OUT []: Zero_weight is 2 (4, 6, 8, and 10 have the same results)
# OUT []: Predicted    1.0    All
# OUT []: True                   
# OUT []: 0.0         6143   6143
# OUT []: 1.0        45712  45712
# OUT []: All        51855  51855
# OUT []: The MODELED accuracy score of the test/valdiation set is 88.2%
# OUT []: The MODELED accuracy on predicted good loans of test/valid. set is 88.154% with 0.000% reduced coverage
# OUT []: The ACTUAL accuracy score of the test/validation set is 88.2%
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.079%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 11.079%
                             
                             
                             
                             
##########################   MODEL 4 - USING ONLY GRADE 'E','F','G' ##########################
dfEFG_rf = df_rf[df_rf['grade'].isin([4,5,6])].copy()
dfEFG_rf.drop('grade', axis=1, inplace=True)

var2 = ['loan_amnt', 'int_rate', 'installment', 'sub_grade',
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'pymnt_plan', 'purpose', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
        'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
        'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal','revol_util', 
        'total_acc', 'initial_list_status', 'fico_range', 'Avg_Median', 'Min_Median', 
        'Pop', 'Dif_mean_median', 'Median_range','Dif_median_from_zip', 'loan_over_income', 
        'loan_over_median', 'RU_Cat', 'RU_Ratio']

### SPLITTING DATA INTO TRAINING AND TESTING SETS
np.random.seed(2016)
msk = np.random.rand(len(dfEFG_rf)) < 0.70
dfEFG_rf_train = dfEFG_rf[msk]
dfEFG_rf_test = dfEFG_rf[~msk]

## Factorized the target, note that sort must be equal True to rain 1 and 0 integrity
y_EFG_train, _ = pd.factorize(dfEFG_rf_train['target'], sort=True)
y_EFG_test, _ = pd.factorize(dfEFG_rf_test['target'], sort=True)

## Converting df_rf dataframe into predictor arrays
X_EFG_train = dfEFG_rf_train.values
X_EFG_test = dfEFG_rf_test.values

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_EFG_train==1) / np.count_nonzero(y_EFG_train==0)
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_EFG_train, osy_EFG_train = OS.fit_transform(X_EFG_train, y_EFG_train)
osX_EFG_test, osy_EFG_test = OS.fit_transform(X_EFG_test, y_EFG_test)

## Converting array back to dataframe.
columns = dfEFG_rf.columns
osdf_EFG_rf_train = pd.DataFrame(data=osX_EFG_train, columns=columns)
osdf_EFG_rf_test = pd.DataFrame(data=osX_EFG_test, columns=columns)

y_pred = random_forrest (osdf_EFG_rf_train[var2], osdf_EFG_rf_test[var2], osdf_EFG_rf_train['target'], osdf_EFG_rf_test['target'], n_jobs=-1, n_est=500, w_start=False, zero_weight=1)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 11.974%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 12.047%

y_pred = random_forrest (osdf_EFG_rf_train[var2], dfEFG_rf_test[var2], osdf_EFG_rf_train['target'], dfEFG_rf_test['target'], n_jobs=-1, n_est=500, w_start=True, zero_weight=1)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 16.138%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.223%

y_pred = random_forrest (dfEFG_rf_train[var2], dfEFG_rf_test[var2], dfEFG_rf_train['target'], dfEFG_rf_test['target'], n_jobs=-1, n_est=500, w_start=False, zero_weight=2)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 16.211%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 16.223%

##########################       MODEL 4 - USING ONLY GRADE 'D'     ##########################
dfD_rf = df_rf[df_rf['grade'] == 3 ].copy()
dfD_rf.drop('grade', axis=1, inplace=True)

### SPLITTING DATA INTO TRAINING AND TESTING SETS
np.random.seed(2016)
msk = np.random.rand(len(dfD_rf)) < 0.70
dfD_rf_train = dfD_rf[msk]
dfD_rf_test = dfD_rf[~msk]

## Factorized the target, note that sort must be equal True to rain 1 and 0 integrity
y_D_train, _ = pd.factorize(dfD_rf_train['target'], sort=True)
y_D_test, _ = pd.factorize(dfD_rf_test['target'], sort=True)

## Converting df_rf dataframe into predictor arrays
X_D_train = dfD_rf_train.values
X_D_test = dfD_rf_test.values

## Balancing the dataset so that good loans vs bad loans volumns are balanced
t_ratio =  np.count_nonzero(y_D_train==1) / np.count_nonzero(y_D_train==0)
OS = OverSampler(ratio=t_ratio, verbose=verbose)
osX_D_train, osy_D_train = OS.fit_transform(X_D_train, y_D_train)
osX_D_test, osy_D_test = OS.fit_transform(X_D_test, y_D_test)

## Converting array back to dataframe.
columns = dfD_rf.columns
osdf_D_rf_train = pd.DataFrame(data=osX_D_train, columns=columns)
osdf_D_rf_test = pd.DataFrame(data=osX_D_test, columns=columns)

y_pred = random_forrest (osdf_D_rf_train[var2], osdf_D_rf_test[var2], osdf_D_rf_train['target'], osdf_D_rf_test['target'], n_jobs=-1, n_est=500, w_start=False, zero_weight=1)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 9.925%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 9.945%

y_pred = random_forrest (osdf_D_rf_train[var2], dfD_rf_test[var2], osdf_D_rf_train['target'], dfD_rf_test['target'], n_jobs=-1, n_est=500, w_start=True, zero_weight=1)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 14.411%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.443%

y_pred = random_forrest (dfD_rf_train[var2], dfD_rf_test[var2], dfD_rf_train['target'], dfD_rf_test['target'], n_jobs=-1, n_est=500, w_start=False, zero_weight=2)
# OUT []: The PREDICTED Annualized ROI of test/validation set on predicted good loans is: 14.443%
# OUT []: The ACTUAL Annualized ROI of test/validation set on overall true good loans is: 14.443%

#--------------------------------------------------------------------------------------------#
##################                  CONCLUDING NOTES:                      ###################
##################     Pretty much the same results as before.             ###################
#--------------------------------------------------------------------------------------------#

'''FOR REFERENCE: 

## LOOK UP COEFFICIENT SIGNIFICANT BASED ON EACH PREDICTORS FOR GRADE SUBGROUPS
df_num_test = df_A.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_A['target'], df_num_test)
result = logit.fit()
print (result.summary())

df_num_test = df_B.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_B['target'], df_num_test)
result = logit.fit()
print (result.summary())

df_num_test = df_C.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_C['target'], df_num_test)
result = logit.fit()
print (result.summary())

df_num_test = df_D.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_D['target'], df_num_test)
result = logit.fit()
print (result.summary())

df_num_test = df_E.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_E['target'], df_num_test)
result = logit.fit()
print (result.summary())

df_num_test = df_FG.get(num_predictors_test)
df_num_test['Intercept'] = 1.0
logit = sm.Logit(df_FG['target'], df_num_test)
result = logit.fit()
print (result.summary())
'''

'''FOR REFERENCE - PART F: LOGISTIC REGRESSION MODEL 1
## Originally, we used all of the numeric predictors in our logistic regression; however,
## the result was dissapointed as the default regression classified all test data as
## "good loans" as if there were no bad loans.  Therefore, we looked up the regression
## coefficients and eliminated the predictors with really low coefficients - see "zipped".
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'annual_inc', \
                  'dti', 'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', \
                  'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'fico_range', \
                  'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',\
                  'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Ratio']
df_num.columns.values
lr.coef_[0]
zipped = zip(df_num.columns.values, lr.coef_[0])
print(list(zipped))

## FOR REFERENCE - THE VERY FIRST DS_NUM SELECTIONS purely based on coefficient values
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'dti', 'fico_range_high', \
                  'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', \
                  'open_acc', 'revol_util', 'total_acc','Dif_median_from_zip', 'RU_Ratio']
                  
## FOR REFERENCE - THE SECOND DS_NUM SELECTIONS based on p-values - THIS IS THE CURRENT SELECTIONS
num_predictors = ['loan_amnt', 'int_rate', 'installment', 'emp_length', 'dti', \
                  'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', \
                  'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', \
                  'total_acc', 'Min_Median', 'Dif_mean_median', 'loan_over_income', 'RU_Ratio']

## FOR REFERENCE: SOME OBSERVATIONS                  
plt.hist(df['Dif_mean_median'])   # histogram plot for reference
df.boxplot(column = 'revol_bal')  # boxplot for reference
# scatterplot for reference - limited the boundaries of y-axis here
plt.ylim([-150000, 200000])
plt.scatter(df['target'], df['Dif_mean_median'])

'''

'''FOR REFERENCE - PART G: LOGISTIC REGRESSION MODEL 2
## FOR REFERENCE - this would be the full varialbes logistic predictor
logistic_predictors_full = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'purpose', 'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'initial_list_status', 'fico_range',
       'Avg_Median', 'Min_Median', 'Pop', 'Dif_mean_median', 'Median_range',
       'Dif_median_from_zip', 'loan_over_income', 'loan_over_median', 'RU_Cat', 'RU_Ratio']
'''