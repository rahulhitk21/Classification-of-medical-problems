# Classification-of-medical-problems

This particular problem’s objective was to determine the primary and the secondary problems from a
text body provided by the customers. Now this there are 4 python files each has its own purpose. The
analysis of the data and the prediction is explained with the description of each of the files.

1. train.py

This file does all the pre-processing steps needed to convert the data file provided into a format
where models can be applied. The particular file provided as a training dataset
(‘lybrate_ml_test.csv’) contained 3 columns body, primary_complains, seconadry_complains. Both
of the columns except body had more than one problem as well as unwanted characters “u” and the
“ ‘’ ” in each of the problem mentioned. Now as our objective is to identify problems among all
the problems mentioned in the provided file we need to convert it into a dataframe where each of
the problems is a column. If a particular problem is present for a particular row than the column
value or that particular row is 1 otherwise 0. So in lybrrate_train.py we remove all the unwanted
characters all convert it into a dataframe where the columns are each problem with the column
named ‘body’ in the original dataset. We save this particular datarame as csv file for further
processing as ‘trans_final_data_local_afterpm1.csv’ as the dataset is huge and otherwise this costly
operation will run every time we wanted to predict problems rom new data.

2. importing.py

After the data converted to the desired format it is a multiclass classification problem. So some pre-
processing needed to be done on the feature variable of the problem which is our ‘body’ column of
the dataset. Now a separation of training and test data has been done here to test the final model.
The last 30 rows considered as testing dataset as test_train_split from sklearn tend to give a memory
error. Now here the punctuations and unwanted characters were removed from the feature and it
was transformed using tfidf vectorizer with n-gram= (1, 2). Now this file was separated from the
modelling file because at the time of testing it needed to be imported on the main file and if we
include it into the modelling file(where training has been done) unwanted training will run again
which will consume a lot of time.

3. modelling.py

Now as the preprocessing was done the training data was fed into the model which is a logistic
regression model coupled with naïve bayes . This model showed good performance in the past as
suggested by Wang et. al. We saved this particular model as a pickle file to further use it for final
prediction.

4. lybrate_main.py

This is the main file where all the predictions are being made. Now it takes the path of the target
dataset from the command line argument by this command ‘Python lybrate_main.py --Data_File
C:\Users\Rahul\Desktop\edwisor\test_data.csv’ where ‘C:\Users\Rahul\Desktop\edwisor’ is the
path to the target file. Now importing.py was imported into this module also and transformation
was done on the ‘body’ column of the target dataset to feed it into the model. The particular model
was loaded in this file via joblib.load() and target dataset was fed into it to predict the probabilities
of having a problem among all the problems previously mentioned. Based on this probability values
the problems were tagged as primary and secondary and the predictions were saved into
‘final_report.csv’ file. Now the threshold value of detecting a secondary problem was purposefully
kept very low value because as this a health problem, overlooking of a problem might have serious
effect on the health of the patient
