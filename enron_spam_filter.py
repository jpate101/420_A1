#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# The dataset contains a total of 17.171 spam and 16.545 non-spam ("ham") e-mail messages (33.716 e-mails total). 

import pandas as pd
import dask.dataframe as dd
import csv
import os
import shutil
import re
import numpy as np

train = pd.read_csv("./data/train_data.zip", compression="zip", index_col="Message ID")
train.head()

# Column	Explanation
# Subject	The subject line of the e-mail
# Message	The content of the e-mail. Can contain an empty string if the message had only a subject line and no body. In case of forwarded emails or replies, this also contains the original message with subject line, "from:", "to:", etc.
# Spam/Ham	Has the values "spam" or "ham". Whether the message was categorized as a spam message or not.
# Date	The date the e-mail arrived. Has a YYYY-MM-DD format.

print("Total Count")
print(train["Spam/Ham"].value_counts(dropna=False))
print("\nProportion in %")
print(round(train["Spam/Ham"].value_counts(normalize=True), 4)*100)

# read in subject vocabulary dataframe with dask 
if not os.path.exists("./data/subject_voc.csv"):
    shutil.unpack_archive("./data/subject_voc.csv", "data/")
subject_voc = dd.read_csv("./data/subject_voc.csv").set_index("Message ID")

# =============================================================================
# Building the Spam Filter
# General Constant Parameters for Naive Bayes
# =============================================================================
# Calculate probability for Spam and Non-Spam (Ham)
p_spam = train["Spam/Ham"].value_counts(normalize=True)["spam"]
p_ham = train["Spam/Ham"].value_counts(normalize=True)["ham"]

# Spam Filter based on Subject Line
# Calculating Subject Line Specific Constant Parameters
# Word Counts for Spam/ham
train_spam = train[train["Spam/Ham"] == "spam"]
train_ham = train[train["Spam/Ham"] == "ham"]

n_words_subject_spam = train_spam["Subject"].apply(len).sum()
n_words_subject_ham = train_ham["Subject"].apply(len).sum()
# (could also use train[mask_spam].iloc[:,2:].sum().sum() above, but takes approx 2x as long)

# Unique word count for vocabulary
n_words_subject_voc = subject_voc.shape[1]

# Smoothin parameter
alpha = 1

# Delete train_spam and train_ham again to save memory
del train_spam, train_ham

# only compute word-specific probabilities if not already saved to file
if not (os.path.exists("data/p_subject_word_given_spam.csv") and os.path.exists("data/p_subject_word_given_ham.csv")):
    '''
    Doing the following computation from dask dataframes is extremly slow.
    Instead I will create a series of pandas dataframes from the main dask dataframe to perform the operation on
    the result will be stored on file (just in case).
    For this dataset, pandas dataframes with 2500 columns seem to work without memory problems on my machine
    Therefore the following process will go through the bigger dask dataframe with smaller pandas dataframe, 
    each containing 2500 columns.
    '''

    print("No files with probabilities for subject line words found. Creating dict with probabilities given spam/ham for " +
          str(n_words_subject_voc) + " words...")

    # Build dictionaries with word-specific probability given either spam or non-spam (ham)
    p_subject_word_given_spam_dict = {word: 0.0 for word in subject_voc.columns}
    p_subject_word_given_ham_dict = {word: 0.0 for word in subject_voc.columns}

    # Determine slice endpoints to go through dask dataframe in 2500 word steps
    endpoints = list(range(2500, n_words_subject_voc, 2500))
    endpoints.append(n_words_subject_voc)

    step = 1

    for endpoint in endpoints:
        print("creating dictonary - step " + str(step) + "/" +
              str(len(endpoints)) + "...", end="\r")

        subject_voc_words_step = list(subject_voc.columns[:endpoint])

        # Limit subject vocabulary dataframe to the 2500 words in this step, 
        # Add Spam/Ham to dataframe
        # Seperate dataframe into spam/ham dataframes
        # Then transform from dask to pandas dataframe
        subject_voc_spam = subject_voc[subject_voc_words_step].copy()
        subject_voc_ham = subject_voc[subject_voc_words_step].copy()
        subject_voc_spam["Spam/Ham"] = train["Spam/Ham"]
        subject_voc_ham["Spam/Ham"] = train["Spam/Ham"]
        subject_voc_spam = subject_voc_spam[subject_voc_ham["Spam/Ham"] == "spam"]
        subject_voc_ham = subject_voc_ham[subject_voc_ham["Spam/Ham"] == "ham"]
        subject_voc_spam = subject_voc_spam.compute()
        subject_voc_ham = subject_voc_ham.compute()

        for word in subject_voc_words_step:
            n_word_given_spam = subject_voc_spam[word].sum()
            prob = (n_word_given_spam + alpha) / \
                (n_words_subject_spam + alpha * n_words_subject_voc)
            p_subject_word_given_spam_dict[word] = prob

            n_word_given_ham = subject_voc_ham[word].sum()
            prob = (n_word_given_ham + alpha) / \
                (n_words_subject_ham + alpha * n_words_subject_voc)
            p_subject_word_given_ham_dict[word] = prob

        step += 1

    print("dictonary created")

    print("Now saving dictonaries with probabilities to file...")

    with open("data/p_subject_word_given_spam.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, p_subject_word_given_spam_dict.keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerow(p_subject_word_given_spam_dict)

    with open("data/p_subject_word_given_ham.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, p_subject_word_given_ham_dict.keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerow(p_subject_word_given_ham_dict)

    print("Csv-files saved to 'data/p_subject_word_given_spam.csv' and 'data/p_subject_word_given_ham.csv'")
# if word-specific probabilities have already been save to file, read them in and store in dictonary
else:
    print("Files with probabilities for subject line words found. Reading in probabilities to dictonaries.", end = "")
    with open("data/p_subject_word_given_spam.csv") as f:
        reader = csv.DictReader(f)
        p_subject_word_given_spam_dict = dict(next(reader))
    
    with open("data/p_subject_word_given_ham.csv") as f:
        reader = csv.DictReader(f)
        p_subject_word_given_ham_dict = dict(next(reader))
    print("Finished!")

# Build Spam Classification Function
# The following function takes a subject line (as a string) and classifies the subject 
# line as ham or spam based on the unnormalized posterior, computed from the parameters
#  derived from the training dataset. I do not normalize the posterior, because normalizing 
#  does not matter for the classification so we can avoid an extra computational step. 
#  By default the function returns a pandas series with a string classifying the message ("ham"/"spam"/"unsure") 
#  and the unnormalized posterior for ham and spam. The function also includes a 
#  test_mode (if test_mode = True). In test mode the function does not have return values, 
#  but instead prints the unnormalized posterior and the classification to the terminal.

def classify_spam_subject(subject, test_mode = False):
    '''
    Takes a string subject line and classifies it as spam or ham
    
    test_mode = False returns a pandas series with a string classifying the message ("ham"/"spam"/"unsure")
                and the unnormalized posterior for ham and spam
    test_mode = True prints out a message to terminal with probabilities and classification
    '''

    # Remove all punctuation and convert everything to lowercase
    message = re.sub('\W', ' ', subject).lower()
    # The method above is quick but produces double spaces - cleaning this up
    message = re.sub('\s{2,}', ' ', subject).strip()

    # Convert message to list
    word_list = subject.split(" ")
    
    # intialize posteriors
    p_spam_given_subject = np.float64(p_spam)
    p_ham_given_subject = np.float64(p_ham)
    
    for word in word_list:
        if word in p_subject_word_given_spam_dict:
            # (approx. 2x faster than "in voc", but p_word_given_spam_dict and p_word_given_ham_dict have the same keys)
            p_spam_given_subject *= np.float64(p_subject_word_given_spam_dict[word])
            p_ham_given_subject *= np.float64(p_subject_word_given_ham_dict[word])
    
    # normalize posteriors
    
    if test_mode:
        print('Unnormalized Posterior (Spam|message):', p_spam_given_subject)
        print('Unnormalized Posterior (Ham|message):', p_ham_given_subject)

        if p_ham_given_subject > p_spam_given_subject:
            print('Label: Ham')
        elif p_ham_given_subject < p_spam_given_subject:
            print('Label: Spam')
        else:
            print('Equal proabilities, have a human classify this.')
    else:
        if p_ham_given_subject > p_spam_given_subject:
            return pd.Series(["ham", p_ham_given_subject, p_spam_given_subject])
        elif p_ham_given_subject < p_spam_given_subject:
            return pd.Series(["spam", p_ham_given_subject, p_spam_given_subject])
        else:
            return pd.Series(["unsure", p_ham_given_subject, p_spam_given_subject])

# Below I will test this function with a random ham & spam message taken from my own e-mail account...

# Test the Spam function
print("Classify message:\n'Spatial Interpolation With and Without Predictor(s) plus 9 more'\n")
classify_spam_subject("Spatial Interpolation With and Without Predictor(s) plus 9 more", test_mode = True)
print("\n\nClassify message:\n'YOU HAVE SUCCESSFULY RECEIVED💲46.527,81 USD INTO YOUR ACCOUNT ⚠️CONFIRM BEFORE ⛔DELETE⛔ IN 48H⚠️'\n")
classify_spam_subject('YOU HAVE SUCCESSFULY RECEIVED💲46.527,81 USD INTO YOUR ACCOUNT ⚠️CONFIRM BEFORE ⛔DELETE⛔ IN 48H⚠️', test_mode = True)

### Testing Classification Function on Test Data Set

# Read in test data
test = pd.read_csv("./data/test_data.zip", compression="zip", index_col="Message ID")

# Use Algorithm to determine p(Ham), p(Spam) and predicted categorization
test[["Predicted", "unnorm post(Ham)", "unnorm post(Spam)"]] = test['Subject'].apply(classify_spam_subject)
# Record correct/incorrect categorizations
test["Correct Categorization"] = (test["Predicted"]==test["Spam/Ham"])
# Calculate p(Ham)/p(Ham) Ratio to determine edge cases
test["Ham/Spam"] = test["unnorm post(Ham)"]/test["unnorm post(Spam)"]
test.head()

## Evaluate Performance of Classification by Subject Line
total_correct = 0
spam_correct = 0
ham_correct = 0
total_messages = test.shape[0]
spam_messages = test[test["Spam/Ham"]=="spam"].shape[0]
predicted_spam = test[test["Predicted"]=="spam"].shape[0]
total_correct = test[test["Correct Categorization"]==True].shape[0]
spam_correct = test[(test["Spam/Ham"]=="spam") & (test["Correct Categorization"]==True)].shape[0]
        
accuracy = total_correct / total_messages
precision = spam_correct / predicted_spam
sensitivity = spam_correct / spam_messages

print("Accuracy of Spam Classification\n(what percentage of all messages is classified correctly)")
print(str(round(accuracy * 100, 2)) + " %")

print("\nPrecision of Spam Classification \n(what percentage of spam classifications is correct)")
print(str(round(precision * 100, 2)) + " %")

print("\nSensitivity of Spam Classification \n(what percentage of actual spam do we classify)")
print(str(round(sensitivity * 100, 2)) + " %")

print("\nF1 Score:")
print(2*(sensitivity * precision) / (sensitivity + precision))

# Even though the algorithm so far only looked at the subject line it detects ***all*** spam emails in the test dataset!
# The precision is also remarkably high with 96%. This means only 4% of "ham" messages are incorrectly classified as spam.
# This example clearly shows that a simple algorithm like naive Bayes can achieve very accurate results given enough data.

# Things to do to improve the algorithm and/or efficiency (might try these things at a later point):
# * Ignore words with low specificity (p(spam) around 0.5) to make algorithm run faster
# * Use p(Ham)/p(Spam) ratio to detect edge cases with unsure classification. In first test runs 0.1 < \[p(Ham)/p(Spam) ratio\] < 10 
# seems to work well to catch cases with unreliable classification. Edge cases could then be run through a more 
# computational intensive algorithm analysing the email text body.

