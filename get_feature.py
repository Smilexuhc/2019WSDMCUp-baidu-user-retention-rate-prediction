
# coding: utf-8

# In[ ]:


from feature_extraction import *

import pandas as pd
import numpy as np
import os
import datetime,time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc


path = "./"
train_file = "train"
test_file = "test"

train_cols = ['UserId', 'gender', 'age', 'edu', 'location',
              "label",
             'install_channel', 'video_id', 'video_type', 'video_tag', 'video_creator',
             'launch_time', 'video_length', 'show', 'click', 'rec_type',
             'play_time_length', 'play_stamp', 'comment', 'like', 'share']

test_cols = ['UserId', 'gender', 'age', 'edu', 'location',
             'install_channel', 'video_id', 'video_type', 'video_tag', 'video_creator',
             'launch_time', 'video_length', 'show', 'click', 'rec_type',
             'play_time_length', 'play_stamp', 'comment', 'like', 'share']

train = load_data(path+train_file, train_cols)
test = load_data(path+test_file, test_cols)


# In[ ]:


train = decode_timestamp(train)
train_user_df = get_train_user_df(train)
train = train.drop(['label'], axis=1)

label = train_user_df['label']
label.index = train_user_df["UserId"]
label.to_csv("label_alldata_12_7.csv")

train_user_df.drop(['label'], inplace=True, axis=1)
train = usage_compression(train)
train_user_df = get_action_count_feat(train_user_df, train)
train_user_df = get_videoid_count_feat(train_user_df, train)
train_user_df = get_video_stats_feat(train_user_df, train)
train_user_df = get_time_feat(train_user_df)
train_user_df = get_type_count_feat(train_user_df, train)
train_user_df = get_time_period_feat(train_user_df, train)
train_user_df = get_deeptime_feat(train_user_df, train)
train_user_df = get_time_diff_feat(train_user_df, train)
train_user_df = get_time_interval_feat(train_user_df)
train_user_df = get_video_play_gap_feat(train_user_df, train)
#train_user_df = get_installchannel_feat(train_user_df, train)
train_user_df = get_700sparse_feat(train_user_df, train)
train_user_df = get_deep_interest_feat(train_user_df, train)
train_user_df.to_csv("train_all_data_notag_12_7.csv")


# In[ ]:


test = decode_timestamp(test)
test_user_df = get_test_user_df(test)
test = usage_compression(test)
test_user_df = get_action_count_feat(test_user_df, test)
test_user_df = get_videoid_count_feat(test_user_df, test)
test_user_df = get_video_stats_feat(test_user_df, test)
test_user_df = get_time_feat(test_user_df)
test_user_df = get_type_count_feat(test_user_df, test)
test_user_df = get_time_period_feat(test_user_df, test)
test_user_df = get_deeptime_feat(test_user_df, test)
test_user_df = get_time_diff_feat(test_user_df, test)
test_user_df = get_time_interval_feat(test_user_df)
test_user_df = get_video_play_gap_feat(test_user_df, test)
#test_user_df = get_installchannel_feat(test_user_df, test)
test_user_df = get_700sparse_feat(test_user_df, test)
test_user_df = get_deep_interest_feat(test_user_df, test)
test_user_df.to_csv("test_all_data_notag_12_7.csv")

