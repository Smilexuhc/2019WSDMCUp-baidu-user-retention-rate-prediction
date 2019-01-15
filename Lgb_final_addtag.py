import pandas as pd
import numpy as np
import os
import datetime,time
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import gc
import time


def label_encode(df, feature):
    print("Start label_encode")
    st = time.time()

    df[feature] = df[feature].replace('-', np.nan)
    le = LabelEncoder()
    for col in feature:
        df[col] = le.fit_transform(df[col].astype(str))

    print('END label_encode', (time.time()-st), 's')
    return df


def get_interval_ratio_feat(user_df):
    interval_cols = ['show_count_all','show_count', 'click_count', 'comment_count','like_count','share_count', 'video_id_sum',
                     'video_tag_sum','video_creator_sum','diff_time_play_count_x','diff_time_show_count_x','diff_time_click_count_x','diff_time_like_count_x',
                     'time_play_count','time_show_count', 'time_click_count','time_like_count','replay_creator_count','reshow_creator_count', 'reclick_creator_count',
                     'replay_video_count', 'reshow_video_count','reclick_video_count', 'replay_type_count','reshow_type_count',
                     'reclick_type_count','diff_time_play_count_y','diff_time_show_count_y', 'diff_time_click_count_y',
                     'diff_time_like_count_y', 'fav_play_type_count','fav_show_type_count', 'fav_click_type_count','fav_like_type_count','interval_sum']
    for col in interval_cols:
        user_df[col + '_interval_cnt_ratio'] = user_df[col] / user_df['interval_cnt']
        user_df[col + '_interval_sum_ratio'] = user_df[col] / user_df['interval_sum']
        user_df[col + '_interval_max_ratio'] = user_df[col] / user_df['interval_max']
        user_df[col + '_interval_min_ratio'] = user_df[col] / user_df['interval_min']
        user_df[col + '_interval_mean_ratio'] = user_df[col] / user_df['interval_mean']
        user_df[col + '_interval_median_ratio'] = user_df[col] / user_df['interval_median']
        user_df[col + '_interval_diff_ratio'] = user_df[col] / user_df['interval_diff']

    return user_df

path = 'D:/Projects/MachineLearning/Data competition/Tabular/wsdm-biaduhaokan/data/'

train = pd.read_csv(path+'train_all_notag.csv')
test = pd.read_csv(path+'test_all_notag.csv')
label = pd.read_csv(path+'all_data_label.csv')
tag = pd.read_csv(path+'').drop(['Unnamed: 0'],axis=1)

full_data = pd.concat([train,test])
del train,test
gc.collect()
full_data = pd.merge(full_data,tag,on=['UserId'],how='left')
del tag
train = full_data[full_data['train']==1].drop(['train'],axis=1)
test = full_data[full_data['train']==0].drop(['train'],axis=1)

test_id = test['UserId']
train = train.drop(['UserId'],axis=1)
test = test.drop(['UserId'],axis=1)

y = label['label'].reset_index().drop(['index'],axis=1)

col = ['fav_click_type','fav_like_type','fav_play_type','fav_show_type']
train = label_encode(train,col)
test = label_encode(test,col)

train = get_interval_ratio_feat(train)
test = get_interval_ratio_feat(test)

cate_feature = ['gender','age','edu','play_mday','play_weekday','play_isweekend','fav_click_type','fav_like_type','fav_play_type','fav_show_type']
feature = list(train.columns)

lgb_model = LGBMClassifier(
    boosting_type="gbdt",num_leaves=64, reg_alpha=3, reg_lambda=3,
    max_depth=-1, n_estimators=10000,
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.01, random_state=1230,n_jobs=-1,
)
predict_result = pd.DataFrame()
predict_result['userid'] = test_id
predict_result['rentention_rate'] = 0
best_score = []
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
st = time.time()
for index, (train_index, test_index) in enumerate(skf.split(train,y)):
    print('Start',index+1,' Fold')
    train_x, test_x, train_y, test_y = train.loc[train_index], train.loc[test_index], y.loc[train_index], y.loc[test_index]
    eval_set = [(train_x, train_y), (test_x, test_y)]
    lgb_model.fit(train_x, train_y, eval_set=eval_set, eval_metric='auc', categorical_feature=cate_feature,
                  early_stopping_rounds=100)
    fi = pd.DataFrame()
    fi['fi'] = lgb_model.feature_importances_
    fi['key'] = feature
    print(fi.sort_values('fi'))
    best_score.append(lgb_model.best_score_['valid_1']['auc'])
    print(best_score)
    print('END',index+1,' Fold',(time.time() - st), 's')
    test_pred = lgb_model.predict_proba(test,
                                        num_iteration=lgb_model.best_iteration_)[:, 1]
    predict_result['rentention_rate'] = predict_result['rentention_rate'] + test_pred

predict_result['rentention_rate'] = predict_result['rentention_rate'] / 5
score_mean = np.mean(best_score)
print(score_mean)
mean = predict_result['rentention_rate'].mean()
print('mean:', mean)

predict_result[['userid', 'rentention_rate']].to_csv(path+'lgb_notag%s.csv' % score_mean, index=False,header=0,float_format='%.4f')