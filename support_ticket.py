# expected file: 
# - tuned_lr.pkl, 
# - [topic_lda, topic_lda.id2word] to get topic information (actually need turn it into df and load)

# import packages
import os
from pickle import load
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

# load topic_csv
topic_df = pd.read_csv(os.path.join(os.getcwd(),'data',"lda_topic.csv"))
print(topic_df)


# load tuned_lr.pkl
with open(os.path.join(os.getcwd(),'data',"tuned_lr.pkl"), "rb") as f:
    clf = load(f)

# Financial Domain Complaint Ticketing System

# try using generative AI to generate some (3-5) financial complaint tickets

# enter title [short_text]
# enter description [long_text]

# preprocess the description
# - nlp, clean, tfidf
# - model.predict() to get ticket category, and priority


# can select which support user/team
# table (sortable and filterable)
# Description | Category | Priority | Resolve (tick to clear)


