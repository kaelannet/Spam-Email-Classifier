#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2.ok')


# # Project 2: Spam/Ham Classification
# ## Feature Engineering, Logistic Regression, Cross Validation
# ## Due Date: Sunday 11/24/19, 11:59PM
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In this project, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this project, you should feel comfortable with the following:
# 
# - Feature engineering with text data
# - Using sklearn libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# ## Warning
# We've tried our best to filter the data for anything blatantly offensive as best as we can, but unfortunately there may still be some examples you may find in poor taste. If you encounter these examples and believe it is inappropriate for students, please let a TA know and we will try to remove it for future semesters. Thanks for your understanding!

# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 1
# 1c | 2
# 2 | 3
# 3a | 2
# 3b | 2
# 4 | 2
# 5 | 2
# 6a | 1
# 6b | 1
# 6c | 2
# 6d | 2
# 6e | 1
# 6f | 3
# 7 | 6
# 8 | 6
# 9 | 3
# 10 | 15
# Total | 55

# # Part I - Initial Analysis

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to Kaggle for evaluation.

# In[3]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[4]:



#Training Set
train_subject_nan = original_training_data['subject'].isna()
original_training_data['subject'] = original_training_data['subject'].fillna("")
print(sum(train_subject_nan))

train_email_nan = original_training_data['subject'].isna()
original_training_data['email'] = original_training_data['email'].fillna("")
print(sum(train_email_nan))

#Test Set
test_subject_nan = test['subject'].isna()
test['subject'] = test['subject'].fillna("")
print(sum(test_subject_nan))

test_email_nan = test['subject'].isna()
test['email'] = test['email'].fillna("")
print(sum(test_email_nan))


# In[5]:


ok.grade("q1a");


# ### Question 1b
# 
# In the cell below, print the text of the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[6]:


first_ham = original_training_data[original_training_data['spam'] == 0]['email'].tolist()[0]
first_spam = original_training_data[original_training_data['spam'] == 1]['email'].tolist()[0]
print(first_ham)
print(first_spam)


# In[7]:


ok.grade("q1b");


# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# Two things that I notice is that the ham message included links to external urls and was formatted like a traditional email. However, the spam email was written in what seems to be html which would allow marketers to more easily format a mass email. 

# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and **validating** the models that we train.  We therefore need to split the training data into separate training and validation datsets.  You will need this **validation data** to assess the performance of your classifier once you are finished training. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[8]:


from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[9]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    
    indicator_array = np.ndarray(shape = (len(texts), len(words))).astype(int)
    for i in range(len(texts)):
        for j in range(len(words)):
            if words[j] in texts[i]:
                indicator_array[i][j] = 1
            else:
                indicator_array[i][j] = 0
    return indicator_array


# In[10]:


ok.grade("q2");


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")
# 
# Hint:
# - You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[11]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row as a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stored in the value column"))
display(df.melt("type"))


# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[12]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts
words = ['love', 'sale', 'family', 'help', 'car']
df = pd.DataFrame(words_in_texts(words, train['email']), columns = words)
df['type'] = train['spam'].replace(1, 'spam')
df['type'] = df['type'].replace(0, 'ham')
new_df = pd.melt(df, 'type')
sns.barplot(new_df['variable'], new_df['value'], hue = new_df['type'], ci = None)
plt.xlabel("Words")
plt.ylabel("Proportion of Emails")


# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 
# ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")
# 

# ### Question 3b
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[13]:


lengths = train.copy()
lengths['lengths'] = lengths['email'].map(lambda x: len(x))

length_ham = lengths[lengths['spam'] == 0]['lengths']
length_spam = lengths[lengths['spam'] == 1]['lengths']
sns.distplot(length_ham, hist = False, label = 'Ham')
sns.distplot(length_spam, hist = False, label = 'Spam')
plt.xlim(0, 50000)
plt.xlabel('Length of email body')
plt.ylabel('Distribution')


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of 0 and 1, no other values). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[14]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[15]:


ok.grade("q4");


# ### Question 5
# 
# Now that we have matrices, we can use to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy around 0.75.
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[16]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_train)
training_accuracy = np.mean(predictions == Y_train)
print("Training Accuracy: ", training_accuracy)


# In[17]:


ok.grade("q5");


# ## Evaluating Classifiers

# That doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe. First, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure, especially if we used the training set to identify discriminative features. In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The following image might help:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# ### Question 6a
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below (answers can be hard-coded):
# 
# *Tests in Question 6 only check that you have assigned appropriate types of values to each response variable, but do not check that your answers are correct.*
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[18]:


zero_predictor_fp = 0
zero_predictor_fn = sum(Y_train)


# In[19]:


ok.grade("q6a");


# ### Question 6b
# 
# What are the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do **NOT** use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# -->

# In[20]:


zero_predictor_acc = 1 - np.mean(Y_train)
zero_predictor_recall = 0
zero_predictor_acc


# In[21]:


ok.grade("q6b");


# ### Question 6c
# 
# Provide brief explanations of the results from 6a and 6b. Why do we observe each of these values (FP, FN, accuracy, recall)?
# 
# <!--
# BEGIN QUESTION
# name: q6c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# -FP = 0 because since our guess is always 0 we can't possibly have a false positive
# 
# -FN = sum(Y_train) because the sum gives us the total number of true elements that were spam and since out guess is ham everytime, we'll have exactly that many false negatives
# 
# -Accuracy = 1 - np.mean(Y_train) because 1 - the mean of Y-train gives us the proportion of ham emails, and since out guess is ham everytime we should be correct by that exact proportion
# 
# -Recall = 0 because it is impossible to get any true positives 
# 
# We observe each of these values because they all represent a different measurement of how well our model is performing:
# FP - Determines how many times we incorrectly predicted a ham email spam (which is important to know because it could result in a user missing an important email because it was flagged as spam)
# FN - Determines how many times we incorrectly predicted a spam email ham (which is important to mitigate in order to prevent copious amounts of spam)
# Accuracy - the proportion of emails were correctly labeled spam/ham
# Recall - For all spam emails the proportion that were correctly labeled spam. Important for identifying how well our model performs on actual spam emails. 

# ### Question 6d
# 
# Compute the precision, recall, and false-alarm rate of the `LogisticRegression` classifier created and trained in Question 5. Do **NOT** use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6d
# points: 2
# -->

# In[22]:


tp = sum((Y_train == 1) & (Y_train == predictions))
fp = sum((Y_train == 0) & (Y_train != predictions))
Y_false = sum(Y_train == 0)
logistic_predictor_precision = tp / sum(predictions)
logistic_predictor_recall = tp / sum(Y_train)
logistic_predictor_far = fp / Y_false
logistic_predictor_precision, logistic_predictor_recall, logistic_predictor_far


# In[23]:


ok.grade("q6d");


# ### Question 6e
# 
# Are there more false positives or false negatives when using the logistic regression classifier from Question 5?
# 
# <!--
# BEGIN QUESTION
# name: q6e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# There are more false negatives than false positives because the logistic regression classifier from question 5 has a recall has a smaller recall than its precision. 

# ### Question 6f
# 
# 1. Our logistic regression classifier got 75.6% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# 1. Given the word features we gave you above, name one reason this classifier is performing poorly. Hint: Think about how prevalent these words are in the email set.
# 1. Which of these two classifiers would you prefer for a spam filter and why? Describe your reasoning and relate it to at least one of the evaluation metrics you have computed so far.
# 
# <!--
# BEGIN QUESTION
# name: q6f
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# (1) It is only slightly more accurate than the zero predictor which had an accuracy of .744
# (2) In terms of the given word features the classifier is performing poorly because the words given aren't very prevalet in the email set so it doesn't provide us with much variability in our data and makes it difficult to effectively catagorize an email correctly using those words. 
# (3) I would prefer to use the classifier developed in question 5 because it has a higher rate of recall meaning that it better flagged spam emails as spam then the zero_predictor. Additionally, one of the benefits of the zero_predictor is that is has a false alarm rate of 0, however the classifier we developed in 5 has a very small false alarm rate.  

# # Part II - Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Kaggle.
# 
# **Kaggle limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on the entire set when submitting to Kaggle (the accuracy that will determine your score for question 10).
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You should use the **validation data** to evaluate your model and get a better sense of how it will perform on the Kaggle evaluation.*
# 
# ---

# In[24]:


#Creation of training set with potential features
email_words = ['<html>', 'sale', 'click', 'money', 'free', 'subscribe', 'please', '!!']
subject_punctuation = ['!', '?', '$']
emails = pd.DataFrame(words_in_texts(email_words, original_training_data['email']), columns = email_words)
emails['bias'] = np.ones(len(emails)).astype(int)
subjects = pd.DataFrame(words_in_texts(subject_punctuation, original_training_data['subject']), columns = subject_punctuation)
X_train = pd.concat([emails, subjects], axis=1, sort=False)
X_train['email_length'] = original_training_data['email'].map(lambda x: len(x))
X_train['subject_length'] = original_training_data['subject'].map(lambda x: len(x))
X_train['subject_caps'] = np.array([sum([1 for c in x if c.isupper()]) for x in original_training_data['subject']])
Y_train = original_training_data['spam']
X_train.head(5)


# In[25]:


#creation of logistic model
spam_model = LogisticRegression()


# In[26]:


#Error for purposes of cross-validation feature selection
def accuracy_test(actual_y, predicted_y):
    """
    Args:
        predicted_y: an array of the prediction from the model
        actual_y: an array of the groudtruth label
        
    Returns:
        The root mean square error between the prediction and the groudtruth
    """
    return sum(actual_y == predicted_y)/len(predicted_y)


# In[27]:


#Cross validation
from sklearn.model_selection import KFold

def compute_CV_accuracy(model, X_train, Y_train):
    '''
    Split the training data into 4 subsets.
    For each subset, 
        fit a model holding out that subset
        compute the MSE on that subset (the validation set)
    You should be fitting 4 models total.
    Return the average MSE of these 4 folds.

    Args:
        model: an sklearn model with fit and predict functions 
        X_train (data_frame): Training data
        Y_train (data_frame): Label 

    Return:
        the average validation MSE for the 4 splits.
    '''
    kf = KFold(n_splits = 3)
    accuracies = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]
        
        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)
        
        # Compute the RMSE on the validation split
        accuracy = accuracy_test(split_Y_valid, model.predict(split_X_valid))
        accuracies.append(accuracy)
        
    return np.mean(accuracies)


# In[28]:


range_of_num_features = range(1, X_train.shape[1] + 1)
accuracies = []
for N in range_of_num_features:
    print(f"Trying first {N} features")
    spam_model = LogisticRegression()
    
    # compute the cross validation error
    accuracy = compute_CV_accuracy(spam_model, X_train.iloc[:, :N], Y_train)
    
    print("\tAccuracy:", accuracy)
    accuracies.append(accuracy)

best_num_features = np.argmax(accuracies) + 1
best_err = max(accuracies)

print(f"Best choice, use the first {best_num_features} features")


# In[29]:


spam_model.fit(X_train, Y_train)
emails = pd.DataFrame(words_in_texts(email_words, test['email']), columns = email_words)
emails['bias'] = np.ones(len(emails)).astype(int)
subjects = pd.DataFrame(words_in_texts(subject_punctuation, test['subject']), columns = subject_punctuation)
X_test = pd.concat([emails, subjects], axis=1, sort=False)
X_test['email_length'] = test['email'].map(lambda x: len(x))
X_test['subject_length'] = test['subject'].map(lambda x: len(x))
X_test['subject_caps'] = np.array([sum([1 for c in x if c.isupper()]) for x in test['subject']])
X_test.head(5)


# ### Question 7: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked / didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q7
# manual: True
# points: 6
# -->
# <!-- EXPORT TO PDF -->

# 1. I analyzed around 10 emails within my own spam inbox on gmail to identify any commonalities in word choice or punctation. This helped me to pick words to analyze within the email and subject bodies. Additionally, I used cross-validation to observe how each additional feature affected the accuracy of my model which resulted in me removing some features and adding others. 
# 2. I tried initially guessing and checking potential word features but realized that it wasn't going to affectively help me see which features were the most effective for my model. However, my use of cross-validation was key for helping me to reduce model variance. 
# 3. I found it surprising that <html> tags were so prevalent within spam emails. Furthermore, there amount of capital letters dramatically increase my accuracy which I didn't expect (I was just playing around with potential features). 

# ### Question 8: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. Include
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel to create as many plots as you want in your process of feature selection, but select one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, as long as it comes with thoughtful commentary. Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q8
# manual: True
# format: image
# points: 6
# -->
# <!-- EXPORT TO PDF format:image -->

# In[30]:


# Write your description (2-3 sentences) as a comment here:
# The heat map plot below helped me to select features because it helps identify the variance in a each feature 
# (for example a word that wasn't very prevalent in any of the emails, low variance should be removed). 
# It also helps to recognize a pattern between each of the features based on the horizontal lines among rows 
# which indicates a correlation between those features and spam/ham emails. 


# Write the code to generate your visualization here:
binary_features = X_train.drop(['bias', 'email_length', 'subject_length', 'subject_caps'], axis = 1)
fig = plt.figure(figsize = (20, 18))
sns.heatmap(binary_features, vmin = 0, vmax = 1)


# In[ ]:





# ### Question 9: ROC Curve
# 
# In most cases we won't be able to get no false positives and no false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover a disease until it's too late to treat, while a false positive means that a patient will probably have to take another screening.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The ROC curve shows this trade off for each possible cutoff probability. In the cell below, plot an ROC curve for your final classifier (the one you use to make predictions for Kaggle). Refer to the Lecture 22 notebook or Section 17.7 of the course text to see how to plot an ROC curve.
# 
# 
# 
# <!--
# BEGIN QUESTION
# name: q9
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# In[31]:


#creating train and 
xtrain, xtest, ytrain, ytest = train_test_split(X_train, original_training_data['spam'], test_size=1000)


# In[32]:


from sklearn.metrics import roc_curve

# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes
spam_model.fit(xtrain, ytrain)
y_percentages = spam_model.predict_proba(xtest)
fpr, tpr, thresh = roc_curve(ytest, y_percentages[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


# # Question 10: Submitting to Kaggle
# 
# The following code will write your predictions on the test dataset to a CSV, which you can submit to Kaggle. You may need to modify it to suit your needs.
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. *Even if you are not submitting to Kaggle, please make sure you've saved your predictions to `test_predictions` as this is how your score for this question will be determined.*
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# You should submit your CSV files to https://www.kaggle.com/c/ds100fa19
# 
# *The provided tests check that your predictions are in the correct format, but you must submit to Kaggle to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q10
# points: 15
# -->

# In[33]:


test_predictions = spam_model.predict(X_test)


# In[34]:


ok.grade("q10");


# The following saves a file to submit to Kaggle.

# In[35]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')


# In[ ]:





# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 9 EXPORTED QUESTIONS -->

# In[ ]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2.ipynb', 'proj2.pdf')
ok.submit()


# In[ ]:




