#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import csv
import zipfile as zf
import os
import sys
import psutil
import requests

while True:
    try:
        user_input = input("Run process with low priority? (y/n): ")
        if user_input.lower() == "y" or user_input.lower() == "yes":
            low_priority = True
            print("Okay - running the process at low priority.")
            break
        elif user_input.lower() == "n" or user_input.lower() == "no":
            low_priority = False
            print("Okay - running the process at normal priority.")
            break
        else:
            raise ValueError  # this will send it to the print message and back to the input option
        break
    except ValueError:
        print("Please enter 'y' or 'n'!")

if low_priority:
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    p = psutil.Process(os.getpid())
    if isWindows:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(1)


print("\nDownloading Enron Spam Data File from Repo (https://github.com/MWiechmann/enron_spam_data)...", end="")
r = requests.get("https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip")
if not os.path.exists("data"):
    os.mkdir("data")

with open("data/enron_spam_data.zip", 'wb') as f:
    f.write(r.content)
print('Done!')

# READ IN DATA #

print("\nReading in data...", end = "")

mails = pd.read_csv("data/enron_spam_data.zip",
                    compression="zip", index_col="Message ID")

print("Done!")
print("\nEmail data read in:")
print("\nTotal:\t" + str(mails.shape[0]))
print(mails["Spam/Ham"].value_counts(dropna=False))

print("\nProportion in %")
print(round(mails["Spam/Ham"].value_counts(normalize=True), 4)*100)

# SETTING UP TRAINING & TESTING DATA SET #

print("\nSeperating Data into training and testing data...", end="")

# Randomize dataset
mails = mails.sample(frac=1, random_state=42)

# Reindex after randomization
mails.reset_index(inplace=True, drop=True)

# Get 80% as training data, rest as test data
cutoff_index = int(round(mails.shape[0] * 0.8, 0))

train = mails.iloc[:cutoff_index].copy(deep=True)
test = mails.iloc[cutoff_index:].copy(deep=True)

# mails is no longer needed - drop from memory
del mails

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

print("Done!")
print("\nCharacteristics of Training & Testing Data Set:")
print("TRAINING DATA:")
print("Proportion in %")
print(round(train["Spam/Ham"].value_counts(normalize=True), 4)*100)

print("\nTESTING DATA:")
print("Proportion in %")
print(round(test["Spam/Ham"].value_counts(normalize=True), 4)*100)

### SAVE TEST DATA AND DROP FROM MEMORY ###
print("\nSaving test data set and dropping it from memory...", end="")

with zf.ZipFile('data/test_data.zip', 'w') as test_zip:
    test_zip.writestr('test_data.csv', test.to_csv(
        index_label="Message ID"), compress_type=zf.ZIP_DEFLATED)

print("DONE!\nTest dataframe saved to 'data/test_data.zip'.")

del test_zip, test

# PREPPING STRING DATA FOR PROCESSING #

print("\nPreparing string data for processing")
print("Transforming strings to lowercase", end ="")

# Prepping Subject Line Data #

# Remove all punctuation and convert everything to lowercase
train["Subject"] = train["Subject"].str.replace(
    "\W", " ", regex=True).str.lower()
# The method above is quick but occasionally produces double spaces - cleaning this up just in case
train["Subject"] = train["Subject"].str.replace(
    "\s{2,}", " ", regex=True).str.strip()

# Prepping message data #
# Remove all punctuation and convert everything to lowercase
train["Message"] = train["Message"].str.replace(
    "\W", " ", regex=True).str.lower()
# The method above is quick but produces double spaces - cleaning this up
train["Message"] = train["Message"].str.replace(
    "\s{2,}", " ", regex=True).str.strip()

# Transform string data to list
print("Done.\nTransforming strings to lists", end="")
train["Subject"] = train["Subject"].str.split(" ")
train["Message"] = train["Message"].str.split(" ")
print("...Done!")


# Saving Dataframes to file
print("\nData Processed. Now saving dataframe to compressed file", end="")

with zf.ZipFile('data/train_data.zip', 'w') as train_zip:
    train_zip.writestr('train_data.csv', train.to_csv(
        index_label="Message ID"), compress_type=zf.ZIP_DEFLATED)

print("Dataframes saved to 'data/train_data.zip'.")
del train_zip

# BUILDING THE VOCABULARY: SUBJECT LINE #

print("\nBuilding the vocabulary: Subject Line...")
print("Collecting set of unqiue words...", end="")
# Build the vocabulary
subject_voc = []

# Add each single word of each message to the vocabulary
for index, subject in train["Subject"].iteritems():
    if type(subject) == list:
        # ignore instance with blank subject lines where the split resulted in nan object instead of list
        for word in subject:
            subject_voc.append(word)

# Get rid of duplicate words
subject_voc = list(set(subject_voc))

print("Done!")
print("\nUnique words in subject line of training data set:")
print(len(subject_voc))

# Build dictonary with word count for each subject line

print("\nCreate dictonary with subject line vocabulary and word count per subject line...", end="")
# Create a dictionary with all unique words as keys and a list as entry
# the list contains one count for each subject line (row) - will be filled in the next step
# because of reindexing ID simply starts at 0...
word_counts_per_subject = {word: [0]*train.shape[0] for word in subject_voc}
word_counts_per_subject["Message ID"] = list(range(train.shape[0]))

# loop over all the subject lines for each contained word
# increase the count in appropriate place in the dictionary by +1

# vals for progress messages

for index, subject in train["Subject"].iteritems():
    for word in subject:
        word_counts_per_subject[word][index] += 1

print("Done!")

# Export data to zip file
print("\nSaving subject line vocabulary to file")

'''
The data is getting a bit large to be handled comfortably on all machines 
(1GB+ for subject line data, memory error for message body data)

Therefore the data will be exported to a csv file so that it can later be read in to a dask dataframe. The uncompressed data might not fit into memory.
Thus, the csv file is first created step by step and then compressed.

'''
def save_dict_to_csv(dictionary, output_file_name, progress_step_size=5):
    print("opening/creating file" + output_file_name + "...")
    with open(output_file_name, 'w', newline='') as csvfile:
        print("writing dictonary to file")
        writer = csv.writer(csvfile)
        writer.writerow(dictionary)  # First row (the keys of the dictionary).
        print("wrote header to file")
        print("now writing values to file")

        dict_vals_zip = zip(*dictionary.values())

        # vals for progress messages
        steps = len(list(dict_vals_zip))
        steps_mult = round(steps / 100)
        current_step = 0
        progress = 0

        for values in zip(*dictionary.values()):
            writer.writerow(values)

            current_step += 1

            if current_step % steps_mult == 0:
                progress += 1
                if progress % progress_step_size == 0:
                    print("about " + str(progress) + "% done", end="\r")

    print("Done!")


print("Saving subject line vocabulary to csv-file")
save_dict_to_csv(word_counts_per_subject, 'data/subject_voc.csv')
print("Now compressing csv-file to zip-file", end="")
zf.ZipFile('data/subject_voc.zip', mode='w').write(
    "data/subject_voc.csv", compress_type=zf.ZIP_DEFLATED)
print("Done!")

del subject_voc, word_counts_per_subject

# BUILDING THE VOCABULARY: MESSAGE BODY #

print("\nBuilding the vocabulary: Messages bodies")
print("Collecting set of unqiue words", end="")

# Build the message_vocabulary
message_voc = []
# Add each single word of each message to the message_vocabulary

for index, message in train["Message"].iteritems():

    if type(message) == list:
        # ignore instance with blank message where the split resulted in nan object instead of list
        for word in message:
            message_voc.append(word)

# Get rid of duplicate words
message_voc = list(set(message_voc))

print("Done!")
print("\nUnique words in message bodies of training data set:")
print(len(message_voc))

print("\ncreate dictonary with message body vocabulary and word count per message")
# Create a dictionary with all unique words as keys and a list as entry
# the list contains one count for each email message (row) - will be filled in the next step
word_counts_per_message = {word: [0]*train.shape[0] for word in message_voc}
# because of reindexing ID simply starts at 0...
word_counts_per_message["Message ID"] = list(range(train.shape[0]))

# loop over all the messages for each contained word
# increase the count in appropriate place in the dictionary by +1

# vals for progress messages
steps = train.shape[0]
steps_mult = round(steps / 100)
progress = 0

for index, subject in train["Message"].iteritems():
    for word in message:
        word_counts_per_message[word][index] += 1

    # print progress
    if index % steps_mult == 0:
        progress += 1
        if progress % 2 == 0:
            print("about " + str(progress) + "% done", end="\r")

print("Done!")

# save message vocabulary to file

print("saving message body vocabulary to csv-file")
save_dict_to_csv(word_counts_per_message,
                 'data/message_voc.csv', progress_step_size=2)
print("Now compressing csv-file to zip-file", end="")
zf.ZipFile('data/message_voc.zip', mode='w').write(
    "data/message_voc.csv", compress_type=zf.ZIP_DEFLATED, compresslevel=9)
print("Finished!\n")

del train, message_voc, word_counts_per_message

while True:
    try:
        user_input = input(
            "Delete uncompressed dictionary files and only keep compressed files? (y/n): ")
        if user_input.lower() == "y" or user_input.lower() == "yes":
            os.remove("data/subject_voc.csv")
            os.remove("data/message_voc.csv")
            print(
                "Okay - uncompressed csv file deleted. All files processed and results saved to file!")
            break
        elif user_input.lower() == "n" or user_input.lower() == "no":
            print(
                "Okay! Keeping uncompressed csv files. All files processed and results saved to file!")
            break
        else:
            raise ValueError  # this will send it to the print message and back to the input option
        break
    except ValueError:
        print("Please enter 'y' or 'n'!")
