import pandas as pd
import gc
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder



train_cols = ['UserId', 'gender', 'age', 'edu', 'location',
              'label', 'install_channel', 'video_id', 'video_type', 'video_tag',
              'video_creator', 'launch_time', 'video_length', 'show', 'click',
              'rec_type', 'play_time_length', 'play_stamp', 'comment', 'like', 'share']
test_cols = ['UserId', 'gender', 'age', 'edu', 'location', 'install_channel', 'video_id',
             'video_type', 'video_tag', 'video_creator', 'launch_time', 'video_length',
             'show', 'click', 'rec_type', 'play_time_length', 'play_stamp', 'comment', 'like', 'share']


def load_data(file, cols, nrows=None):
    # 读取最原始的数据，读取完毕便删除location
    print("Start load data")
    st = time.time()
    if nrows is not None:
        data = pd.read_csv(file, sep='\t', nrows=nrows)
    else:
        data = pd.read_csv(file, sep='\t')
    data.columns = cols

    data.sort_values(by=["UserId", 'play_stamp'], inplace=True)
    data.drop(["location"], axis=1, inplace=True)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}'.
          format(len(data), data.UserId.nunique()))

    print('END load data', (time.time()-st), 's')
    return data


def decode_timestamp(df):
    print("Start decode_timestamp")
    st = time.time()

    time_stamp_play = list(df['play_stamp'] / 1000)
    video_stamp = list(df['launch_time'])
    localtime = []
    for i in time_stamp_play:
        localtime.append(time.localtime(i))
    df['play_mday'] = [localtime[i].tm_mday for i in range(len(localtime))]
    df['play_yday'] = [localtime[i].tm_yday for i in range(len(localtime))]
    df['play_time'] = [localtime[i].tm_hour + localtime[i].tm_min/60 for i in range(len(localtime))]
    del localtime
    localtime_video = []
    for i in video_stamp:
        localtime_video.append(time.localtime(i))
    df['launch_day'] = [localtime_video[i].tm_yday for i in range(len(localtime_video))]
    del localtime_video
    df = df.drop(['play_stamp'], axis=1)
    df = df.drop(['launch_time'], axis=1)
    gc.collect()

    print('END decode_timestamp', (time.time()-st), 's')
    return df


def label_encode(df, feature):
    print("Start label_encode")
    st = time.time()

    df[feature] = df[feature].replace('-', np.nan)
    le = LabelEncoder()
    for col in feature:
        df[col] = le.fit_transform(df[col].astype(str))

    print('END label_encode', (time.time()-st), 's')
    return df


def get_train_user_df(df):
    print("Start get_train_user_df")
    st = time.time()

    user_df = df[['UserId', 'gender', 'age', 'edu', 'label', 'play_mday', 'play_yday'
                  ]].drop_duplicates(subset=['UserId'], keep='first', inplace=False)
    label_encode_feat = ['gender', 'age', 'edu']
    user_df = label_encode(user_df, label_encode_feat)

    print('END get_train_user_df', (time.time()-st), 's')
    return user_df


def get_test_user_df(df):
    print("Start get_test_user_df")
    st = time.time()

    user_df = df[['UserId', 'gender', 'age', 'edu', 'play_mday', 'play_yday'
                  ]].drop_duplicates(subset=['UserId'], keep='first', inplace=False)
    label_encode_feat = ['gender', 'age', 'edu']
    user_df = label_encode(user_df, label_encode_feat)

    print('END get_test_user_df', (time.time()-st), 's')
    return user_df


def usage_compression(df):
    print("Start usage_compression")
    st = time.time()

    # 将user_id列drop（gender，age，edu，label，day_time）
    # 选择drop 列 video_id，video_creator
    # ['show','click','comment','like','share']填充后 转为 int8
    # [video_length,play_time_length]填充后转为 int16和float32
    user_col = ['gender', 'age', 'edu']
    id_col = ['location']
    action_col = ['show', 'click', 'comment', 'like', 'share']
    drop_col = user_col+id_col
    df = df.drop(drop_col, axis=1)
    df[action_col] = df[action_col].replace('-', 0).astype('int8')
    df['video_length'] = df['video_length'].astype('int16')
    df['play_time_length'] = df['play_time_length'].replace('-', 0).astype('float32')
    df['rec_type'] = df['rec_type'].astype('category')

    print('END usage_compression', (time.time()-st), 's')
    return df


def get_action_count_feat(user_df, df):
    print("Start get_action_count_feat")
    st = time.time()

    action_col = ['show', 'click', 'comment', 'like', 'share']
    df[action_col] = df[action_col].replace('-', '0').astype('int8')
    # 在usage 已有处理可注释掉

    user_df = pd.merge(user_df, df['show'].groupby(df['UserId']).count().reset_index().rename(columns={'show':'show_count_all'}), on=['UserId'])
    user_df = pd.merge(user_df, df['show'].groupby(df['UserId']).sum().reset_index().rename(columns={'show':'show_count'}), on=['UserId'])
    user_df = pd.merge(user_df, df['click'].groupby(df['UserId']).sum().reset_index().rename(columns={'click': 'click_count'}), on=['UserId'])
    user_df = pd.merge(user_df, df['comment'].groupby(df['UserId']).sum().reset_index().rename(columns={'comment': 'comment_count'}), on=['UserId'])
    user_df = pd.merge(user_df, df['like'].groupby(df['UserId']).sum().reset_index().rename(columns={'like': 'like_count'}), on=['UserId'])
    user_df = pd.merge(user_df, df['share'].groupby(df['UserId']).sum().reset_index().rename(columns={'share': 'share_count'}), on=['UserId'])

    user_df['show_ratio'] = user_df['show_count']/user_df['show_count_all']
    user_df['click_ratio'] = user_df['click_count']/(user_df['show_count']+1)
    user_df['comment_ratio'] = user_df['comment_count']/(user_df['click_count']+1)
    user_df['like_ratio'] = user_df['like_count']/(user_df['click_count']+1)
    user_df['share_ratio'] = user_df['share_count']/(user_df['click_count']+1)

    print('END get action count feat ', (time.time() - st), 's')
    return user_df


def get_videoid_count_feat(user_df, df):
    # 该函数需在get_action_count_feat后使用
    print("Start get_videoid_count_feat")
    st = time.time()

    video_col = ['video_id', 'video_tag', 'video_creator']
    for col in video_col:
        count = df.groupby([col]).size().to_frame()
        count.columns = [col + '_count']
        df = pd.merge(df, count, on=[col])

        user_df = pd.merge(user_df, df[col + '_count'].groupby(df['UserId']).sum().reset_index().rename(
            columns={col + '_count': col + '_sum'}), on=['UserId'])
        user_df[col + '_ratio'] = user_df[col + '_sum'] / user_df['show_count_all']

    print('END get_videoid_count_feat', (time.time() - st), 's')
    return user_df


def get_video_stats_feat(user_df, df):
    print("Start get_video_stats_feat")
    st = time.time()

    #user_df = pd.merge(user_df, df['Unnamed: 0'].groupby(train['UserId']).mean().reset_index().rename(columns={'Unnamed: 0': 'index_mean'}), on=['UserId'])
    user_df = pd.merge(user_df, df['video_length'].groupby(df['UserId']).mean().reset_index().rename(columns={'video_length': 'video_length_mean'}), on=['UserId'])
    user_df = pd.merge(user_df, df['video_length'].groupby(df['UserId']).max().reset_index().rename(columns={'video_length': 'video_length_max'}), on=['UserId'])
    user_df = pd.merge(user_df, df['video_length'].groupby(df['UserId']).min().reset_index().rename(columns={'video_length': 'video_length_min'}), on=['UserId'])

    # 不知道为啥报错 懒得管了
    # user_df = pd.merge(user_df,df['play_time_length'].groupby(train['UserId']).mean().reset_index().rename(columns={'play_time_length':'play_time_length_mean'}), on=['UserId'])
    # user_df = pd.merge(user_df,df['play_time_length'].groupby(train['UserId']).max().reset_index().rename(columns={'play_time_length':'play_time_length_max'}), on=['UserId'])
    # user_df = pd.merge(user_df,df['play_time_length'].groupby(train['UserId']).min().reset_index().rename(columns={'play_time_length':'play_time_length_min'}), on=['UserId'])

    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).mean().reset_index().rename(columns={'launch_day': 'launch_time_mean'}), on=['UserId'])
    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).max().reset_index().rename(columns={'launch_day': 'launch_time_max'}), on=['UserId'])
    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).min().reset_index().rename(columns={'launch_day': 'launch_time_min'}), on=['UserId'])
    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).std().reset_index().rename(columns={'launch_day': 'launch_time_std'}), on=['UserId'])
    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).skew().reset_index().rename(columns={'launch_day': 'launch_time_skew'}), on=['UserId'])
    user_df = pd.merge(user_df, df['launch_day'].groupby(df['UserId']).apply(lambda x:x.kurt()).reset_index().rename(columns={'launch_time':'launch_time_kurt'}), on=['UserId'])

    # play
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].nunique()).reset_index().rename(columns={0: 'time_play_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].var()).reset_index().rename(columns={0: 'time_play_var'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].mean()).reset_index().rename(columns={0: 'time_play_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].median()).reset_index().rename(columns={0: 'time_play_median'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].mad()).reset_index().rename(columns={0: 'time_play_mad'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].skew()).reset_index().rename(columns={0: 'time_play_skew'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].kurt()).reset_index().rename(columns={0: 'time_play_kurt'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].max()).reset_index().rename(columns={0: 'time_play_max'}), on=['UserId'], how='left')
    #user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_play_max'}), on=['UserId'], how='left')
    # user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().max()-x['play_time'].sort_values().diff().min()).reset_index().rename(columns={0:'time_play_max_gap'}))
    print('Play Done')

    # show
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].nunique()).reset_index().rename(columns={0: 'time_show_count'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].var()).reset_index().rename(columns={0: 'time_show_var'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].mean()).reset_index().rename(columns={0: 'time_show_mean'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].median()).reset_index().rename(columns={0: 'time_show_median'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].mad()).reset_index().rename(columns={0: 'time_show_mad'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].skew()).reset_index().rename(columns={0: 'time_show_skew'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].kurt()).reset_index().rename(columns={0: 'time_show_kurt'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].max()).reset_index().rename(columns={0: 'time_show_max'}), on=['UserId'],how='left')
    #user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_show_max'}), on=['UserId'],how='left')
    print('Show Done')

    # click
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].nunique()).reset_index().rename(columns={0: 'time_click_count'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].var()).reset_index().rename(columns={0: 'time_click_var'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].mean()).reset_index().rename(columns={0: 'time_click_mean'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].median()).reset_index().rename(columns={0: 'time_click_median'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].mad()).reset_index().rename(columns={0: 'time_click_mad'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].skew()).reset_index().rename(columns={0: 'time_click_skew'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].kurt()).reset_index().rename(columns={0: 'time_click_kurt'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].max()).reset_index().rename(columns={0: 'time_click_max'}), on=['UserId'],how='left')
    #user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_click_max'}),on=['UserId'], how='left')
    print('Click Done')

    # like
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].nunique()).reset_index().rename(columns={0: 'time_like_count'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].var()).reset_index().rename(columns={0: 'time_like_var'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].mean()).reset_index().rename(columns={0: 'time_like_mean'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].median()).reset_index().rename(columns={0: 'time_like_median'}),on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].mad()).reset_index().rename(columns={0: 'time_like_mad'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].skew()).reset_index().rename(columns={0: 'time_like_skew'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].kurt()).reset_index().rename(columns={0: 'time_like_kurt'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].max()).reset_index().rename(columns={0: 'time_like_max'}), on=['UserId'],how='left')
    #user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_like_max'}), on=['UserId'], how='left')
    print('Like Done')
    del df
    gc.collect()

    print('END get video stats feat ', (time.time() - st), 's')
    return user_df


def get_tag_feat(user_df, df, top_num):
    print("Start get_tag_feat")
    st = time.time()

    temp = pd.DataFrame(df['video_tag'])
    temp.columns = ['video_tag']
    temp["video_tag_list"] = temp['video_tag'].astype(str).apply(lambda x: x.split('$', x.count('$')))

    print("\tStart get video_tag_top")
    st2 = time.time()
    video_tag_top = []
    for video_tag in temp["video_tag_list"].values:
        for i in video_tag:
            video_tag_top.append(i)

    utt50 = pd.DataFrame(np.array(video_tag_top).reshape(-1, 1))
    utt50.columns = ["video_tag_"]
    top_video_tag = utt50.video_tag_.value_counts().index[: top_num+1]
    print('\tEND get video_tag_top', (time.time()-st2), 's')

    def tag_ohe(data):
        for k, j in enumerate(top_video_tag[1:]):
            data[j] = temp["video_tag_list"].apply(lambda x: 1 if j in x else 0)
            print(k, "-th tag", j+' done')
        return data

    def tag_count(df):
        if df['video_tag'] == np.nan:
            count = 0
        else:
            count = len(str(df['video_tag']).strip().split('$'))
        return count

    print("\tStart get tag_count")
    st2 = time.time()
    df['tag_count'] = df.apply(lambda row: tag_count(row), axis=1)
    print('\tEND get tag_count', (time.time()-st2), 's')

    df = tag_ohe(df)
    for i in top_video_tag:
        user_df = pd.merge(user_df, df[i].groupby(df['UserId']).sum().reset_index().rename(columns={i: i + '_count'}), on=['UserId'])

    print('END get_time_feat', (time.time()-st), 's')
    return user_df


def get_time_feat(user_df):
    print("Start get_tag_feat")
    st = time.time()

    def to_weekday(day):
        if day == 15 or day == 22 or day == 29:
            return 1
        elif day == 16 or day == 23 or day == 30:
            return 2
        elif day == 17 or day == 24 or day == 31:
            return 3
        elif day == 18 or day == 25:
            return 4
        elif day == 19 or day == 26:
            return 5
        elif day == 20 or day == 27:
            return 6
        elif day == 21 or day == 28:
            return 7

    def to_weekend(weekday):
        if weekday == 1 or weekday == 2 or weekday == 3 or weekday == 4 or weekday == 5:
            return 0
        else:
            return 1

    user_df['play_weekday'] = user_df['play_mday'].apply(lambda x: to_weekday(x))
    user_df['play_isweekend'] = user_df['play_weekday'].apply(lambda x: to_weekend(x))

    print('END get_time_feat', (time.time()-st), 's')
    return user_df


def get_type_count_feat(user_df, df):
    print("Start get_type_count_feat")
    st = time.time()

    # 对rec_type 做了统计特征
    user_df = pd.merge(user_df, df['rec_type'].groupby(df['UserId']).value_counts().unstack().fillna(0).rename(columns={'rec_type_1': 'rec_type_1_show0', 'rec_type_2': 'rec_type_2_show0','rec_type_3': 'rec_type_3_show0'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df['rec_type'][df['show'] == 1].groupby(df['UserId'][df['show']==1]).value_counts().unstack().fillna(0). rename(columns={'rec_type_1': 'rec_type_1_show1', 'rec_type_2': 'rec_type_2_show1','rec_type_3': 'rec_type_3_show1'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df,df['rec_type'][df['click'] == 1].groupby(df['UserId'][df['click']==1]).value_counts().unstack().fillna(0).rename(columns = {'rec_type_1': 'rec_type_1_click', 'rec_type_2': 'rec_type_2_click','rec_type_3': 'rec_type_3_click'}),on=['UserId'],how='left')
    #不知道为啥报错
    #user_df = pd.merge(user_df,df['rec_type'][df['comment'] == 1].groupby(df['UserId']).value_counts().unstack().fillna(0).rename(columns = {'rec_type_1': 'rec_type_1_comment', 'rec_type_2': 'rec_type_2_comment','rec_type_3': 'rec_type_3_comment'}),on=['UserId'])
    user_df = pd.merge(user_df,df['rec_type'][df['like'] == 1].groupby(df['UserId'][df['like']==1]).value_counts().unstack().fillna(0).rename(columns = {'rec_type_1': 'rec_type_1_like', 'rec_type_2': 'rec_type_2_like','rec_type_3': 'rec_type_3_like'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df,df['rec_type'][df['share'] == 1].groupby(df['UserId'][df['share']==1]).value_counts().unstack().fillna(0).rename(columns = {'rec_type_1': 'rec_type_1_share', 'rec_type_2': 'rec_type_2_share','rec_type_3': 'rec_type_3_share'}),on=['UserId'],how='left')
    print('END get type count feat ', (time.time() - st), 's')
    return user_df


def get_time_period_feat(user_df, df):
    print("Start get_time_period_feat")
    st = time.time()
    #将每天划分为5个时间段 做click和like share comment的count特征
    def to_dayperiod(day):
        if day>0 and day<6:
            return 5
        elif day >=6 and day <11:
            return 1
        elif day >=11 and day <13.5:
            return 2
        elif day>=13.5 and day <18:
            return 3
        elif day >=18 and day <=21:
            return 4
        elif day>=21 and day<24:
            return 5
        else:
            return np.nan

    df['play_period'] = df['play_time'].apply(to_dayperiod)

    user_df = pd.merge(user_df, df['play_period'][df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).value_counts().unstack().fillna(0).rename(columns={1.0: 'period_show_1', 2.0: 'period_show_2', 3.0: 'period_show_3',4.0: 'period_show_4', 5.0: 'period_show_5'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df['play_period'][df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).value_counts().unstack().fillna(0).rename(columns = {1.0:'period_click_1',2.0:'period_click_2',3.0:'period_click_3',4.0: 'period_click_4',5.0:'period_click_5'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df['play_period'][df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).value_counts().unstack().fillna(0).rename(columns = {1.0:'period_like_1',2.0:'period_like_2',3.0:'period_like_3',4.0: 'period_like_4',5.0:'period_like_5'}),on=['UserId'],how='left')

    print('END get time period feat ', (time.time() - st), 's')
    return user_df


def get_deeptime_feat(user_df, df):
    # 此函数需在usage压缩后使用
    print("Start get_deeptime_feat")
    st = time.time()
    # 没有comment 是因为前100w实在没有人有comment ==1的
    user_df = pd.merge(user_df,df['play_time'].groupby(df['UserId']).apply(lambda x: x.max()-x.min()).reset_index().rename(columns={'play_time': 'play_duration'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df,df['play_time'][df['show'] == 1].groupby(df['UserId']).apply(lambda x: x.max()-x.min()).reset_index().rename(columns={'play_time': 'show_duration'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df['play_time'][df['click'] == 1].groupby(df['UserId']).apply(lambda x: x.max() - x.min()).reset_index().rename(columns={'play_time': 'click_duration'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df['play_time'][df['like'] == 1].groupby(df['UserId']).apply(lambda x: x.max() - x.min()).reset_index().rename(columns={'play_time': 'like_duration'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df['play_time'][df['share'] == 1].groupby(df['UserId']).apply(lambda x: x.max() - x.min()).reset_index().rename(columns={'play_time': 'share_duration'}), on=['UserId'],how='left')

    # 对click过的视频观看率的统计特征
    # 须对‘play_time_length'做.replace('-',0).astype('float32')处理
    user_df = pd.merge(user_df, df[df['click'] == 1].apply(lambda x: x['play_time_length']/(x['video_length']+0.01),axis=1).groupby(
                        df['UserId'][df['click'] == 1]).mean().reset_index().rename(columns = {'0':'watch_ratio_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].apply(lambda x: x['play_time_length']/(x['video_length']+0.01),axis=1).groupby(
                        df['UserId'][df['click'] == 1]).max().reset_index().rename(columns = {'0':'watch_ratio_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].apply(lambda x: x['play_time_length']/(x['video_length']+0.01),axis=1).groupby(
                        df['UserId'][df['click'] == 1]).min().reset_index().rename(columns = {'0':'watch_ratio_min'}),on=['UserId'],how='left')
    #user_df = pd.merge(user_df,df[df['click']==1].apply(lambda x: x['play_time_length'](x['video_length']+0.01),axis=1).groupby(
                        #df['UserId'][df['click']==1]).std().reset_index().rename(columns = {'0':'watch_ratio_std'}),on=['UserId'])
    user_df = pd.merge(user_df, df[df['click'] == 1].apply(lambda x: x['play_time_length']/(x['video_length']+0.01),axis=1).groupby(
                        df['UserId'][df['click'] == 1]).skew().reset_index().rename(columns = {'0':'watch_ratio_skew'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].apply(lambda x: x['play_time_length']/(x['video_length']+0.01),axis=1).groupby(
                        df['UserId'][df['click'] == 1]).apply(lambda x: x.kurt()).reset_index().rename(columns = {'0':'watch_ratio_kurt'}),on=['UserId'],how='left')

    '''
    #不同时间点的count特征
    user_df = pd.merge(user_df,df[['UserId','play_time']].drop_duplicates(subset=['UserId','play_time'],keep='first').groupby(
                        df['UserId']).count().drop(['UserId'],axis=1).rename(columns = {'play_time':'play_time_play_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df,df[['UserId','play_time']][df['click']==1].drop_duplicates(subset=['UserId','play_time'],keep='first').groupby(
        df['UserId'][df['click']==1]).count().drop(['UserId'],axis=1).rename(columns = {'play_time':'play_time_click_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df,df[['UserId', 'play_time']][df['like'] == 1].drop_duplicates(subset=['UserId', 'play_time'],keep='first').groupby(
                           df['UserId'][df['like'] == 1]).count().drop(['UserId'], axis=1).rename(
                           columns={'play_time': 'play_time_like_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df,df[['UserId', 'play_time']][df['show'] == 1].drop_duplicates(subset=['UserId', 'play_time'],keep='first').groupby(
                           df['UserId'][df['show'] == 1]).count().drop(['UserId'], axis=1).rename(
                           columns={'play_time': 'play_time_show_count'}),on=['UserId'],how='left')

    
    '''
    print('END get deep time feat ', (time.time() - st), 's')
    return user_df


def get_time_diff_feat(user_df, df):
    print("Start get_time_diff_feat")
    st = time.time()

    # play
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().nunique()).reset_index().rename(columns={0:'diff_time_play_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().var()).reset_index().rename(columns={0: 'diff_time_play_var'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().mean()).reset_index().rename(columns={0: 'diff_time_play_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().median()).reset_index().rename(columns={0: 'diff_time_play_median'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().mad()).reset_index().rename(columns={0: 'diff_time_play_mad'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().skew()).reset_index().rename(columns={0: 'diff_time_play_skew'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().kurt()).reset_index().rename(columns={0: 'diff_time_play_kurt'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().max()).reset_index().rename(columns={0: 'diff_time_play_max'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_play_max'}), on=['UserId'], how='left')
    # 和一阶差分max一毛一样
    # user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['play_time'].diff().max()-x['play_time'].sort_values().diff().min()).reset_index().rename(columns={0:'diff_time_play_max_gap'}))
    print('Play Done')

    #show
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().nunique()).reset_index().rename(columns={0: 'diff_time_show_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().var()).reset_index().rename(columns={0: 'diff_time_show_var'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().mean()).reset_index().rename(columns={0: 'diff_time_show_mean'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().median()).reset_index().rename(columns={0: 'diff_time_show_median'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().mad()).reset_index().rename(columns={0: 'diff_time_show_mad'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().skew()).reset_index().rename(columns={0: 'diff_time_show_skew'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().kurt()).reset_index().rename(columns={0: 'diff_time_show_kurt'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff().max()).reset_index().rename(columns={0: 'diff_time_show_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_show_max'}),on=['UserId'],how='left')
    print('Show Done')

    # click
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().nunique()).reset_index().rename(columns={0: 'diff_time_click_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().var()).reset_index().rename(columns={0: 'diff_time_click_var'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().mean()).reset_index().rename(columns={0: 'diff_time_click_mean'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().median()).reset_index().rename(columns={0: 'diff_time_click_median'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().mad()).reset_index().rename(columns={0: 'diff_time_click_mad'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().skew()).reset_index().rename(columns={0: 'diff_time_click_skew'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().kurt()).reset_index().rename(columns={0: 'diff_time_click_kurt'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff().max()).reset_index().rename(columns={0: 'diff_time_click_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_click_max'}),on=['UserId'],how='left')
    print('Click Done')

    #like
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().nunique()).reset_index().rename(columns={0: 'diff_time_like_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().var()).reset_index().rename(columns={0: 'diff_time_like_var'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().mean()).reset_index().rename(columns={0: 'diff_time_like_mean'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().median()).reset_index().rename(columns={0: 'diff_time_like_median'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().mad()).reset_index().rename(columns={0: 'diff_time_like_mad'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().skew()).reset_index().rename(columns={0: 'diff_time_like_skew'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().kurt()).reset_index().rename(columns={0: 'diff_time_like_kurt'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff().max()).reset_index().rename(columns={0: 'diff_time_like_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['play_time'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_like_max'}),on=['UserId'],how='left')
    print('Like Done')

    print('END get diff time feat ', (time.time() - st), 's')
    return user_df


def get_time_interval_feat(user_df):
    #在get_diff-timefeat后
    print("Start get_time_interval_feat")
    st = time.time()

    user_df['play_intervel'] = user_df['play_duration'] / user_df['diff_time_play_count']
    user_df['show_intervel'] = user_df['show_duration'] / user_df['diff_time_show_count']
    user_df['click_intervel'] = user_df['click_duration'] / user_df['diff_time_click_count']
    user_df['like_intervel'] = user_df['like_duration'] / user_df['diff_time_like_count']

    print('END get_time_interval_feat ', (time.time() - st), 's')
    return user_df


def get_video_play_gap_feat(user_df,df):
    print('Start get_video_play_time_feat')
    st = time.time()
    df['watch_gap'] = df['play_yday'] - df['launch_day']

    # play
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].nunique()).reset_index().rename(
        columns={0: 'diff_time_play_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].var()).reset_index().rename(
        columns={0: 'diff_time_play_var'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].mean()).reset_index().rename(
        columns={0: 'diff_time_play_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].median()).reset_index().rename(
        columns={0: 'diff_time_play_median'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].mad()).reset_index().rename(
        columns={0: 'diff_time_play_mad'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].skew()).reset_index().rename(
        columns={0: 'diff_time_play_skew'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].kurt()).reset_index().rename(
        columns={0: 'diff_time_play_kurt'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].max()).reset_index().rename(
        columns={0: 'diff_time_play_max'}), on=['UserId'], how='left')
    # user_df = pd.merge(user_df,df.groupby(['UserId']).apply(lambda x: x['watch_gap'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_play_max'}), on=['UserId'], how='left')
    # user_df = pd.merge(user_df, df.groupby(['UserId']).apply(lambda x: x['watch_gap'].diff().max()-x['watch_gap'].sort_values().diff().min()).reset_index().rename(columns={0:'diff_time_play_max_gap'}))
    print('Play Done')

    # show
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].nunique()).reset_index().rename(columns={0: 'diff_time_show_count'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].var()).reset_index().rename(columns={0: 'diff_time_show_var'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].mean()).reset_index().rename(columns={0: 'diff_time_show_mean'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].median()).reset_index().rename(columns={0: 'diff_time_show_median'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].mad()).reset_index().rename(columns={0: 'diff_time_show_mad'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].skew()).reset_index().rename(columns={0: 'diff_time_show_skew'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].kurt()).reset_index().rename(columns={0: 'diff_time_show_kurt'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(
        lambda x: x['watch_gap'].max()).reset_index().rename(columns={0: 'diff_time_show_max'}), on=['UserId'],
                       how='left')
    # user_df = pd.merge(user_df, df[df['show'] == 1].groupby(df['UserId'][df['show'] == 1]).apply(lambda x: x['watch_gap'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_show_max'}), on=['UserId'],how='left')
    print('Show Done')

    # click
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].nunique()).reset_index().rename(columns={0: 'diff_time_click_count'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].var()).reset_index().rename(columns={0: 'diff_time_click_var'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].mean()).reset_index().rename(columns={0: 'diff_time_click_mean'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].median()).reset_index().rename(columns={0: 'diff_time_click_median'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].mad()).reset_index().rename(columns={0: 'diff_time_click_mad'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].skew()).reset_index().rename(columns={0: 'diff_time_click_skew'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].kurt()).reset_index().rename(columns={0: 'diff_time_click_kurt'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(
        lambda x: x['watch_gap'].max()).reset_index().rename(columns={0: 'diff_time_click_max'}), on=['UserId'],
                       how='left')
    # user_df = pd.merge(user_df, df[df['click'] == 1].groupby(df['UserId'][df['click'] == 1]).apply(lambda x: x['watch_gap'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_click_max'}),on=['UserId'], how='left')
    print('Click Done')

    # like
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].nunique()).reset_index().rename(columns={0: 'diff_time_like_count'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].var()).reset_index().rename(columns={0: 'diff_time_like_var'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].mean()).reset_index().rename(columns={0: 'diff_time_like_mean'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].median()).reset_index().rename(columns={0: 'diff_time_like_median'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].mad()).reset_index().rename(columns={0: 'diff_time_like_mad'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].skew()).reset_index().rename(columns={0: 'diff_time_like_skew'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].kurt()).reset_index().rename(columns={0: 'diff_time_like_kurt'}), on=['UserId'],
                       how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(
        lambda x: x['watch_gap'].max()).reset_index().rename(columns={0: 'diff_time_like_max'}), on=['UserId'],
                       how='left')
    # user_df = pd.merge(user_df, df[df['like'] == 1].groupby(df['UserId'][df['like'] == 1]).apply(lambda x: x['watch_gap'].diff(2).max()).reset_index().rename(columns={0: 'diff2_time_like_max'}), on=['UserId'], how='left')
    print('Like Done')

    print('END get_video_watch_gap_feat ', (time.time() - st), 's')
    return user_df


def get_installchannel_feat(user_df, df):
    print("Start get_installchannel_feat")
    st = time.time()

    pd.merge(user_df, df[['UserId', 'install_channel']].drop_duplicates(subset=['UserId', 'install_channel'],
        keep='first').groupby(df['UserId']).count().drop(['UserId'], axis=1)).rename(columns={'install_channel': 'install_channel_count'}, on=['UserId'], how='left')

    print('END get_installchannel_feat ', (time.time() - st), 's')
    return user_df


def get_700sparse_feat(user_df, df,key='click'):
    #放到usage后单独跑就行
    print("Start get 700 sparse_feat")
    st = time.time()
    temp = df[['video_type', 'UserId']][df['click'] == 1]
    user_df = pd.merge(user_df, temp['video_type'].groupby(temp['UserId']).value_counts().unstack().fillna(0), on=['UserId'],how='left')
    del temp
    temp = df[['install_channel', 'UserId']][df['click'] == 1]
    user_df = pd.merge(user_df, temp['install_channel'].groupby(temp['UserId']).value_counts().unstack().fillna(0), on=['UserId'],how='left')
    del temp
    print('END get 700 sparse_feat ', (time.time() - st), 's')
    return user_df


def get_deep_interest_feat(user_df,df):
    print("Start get deep interest feat")
    st = time.time()
    #video_creator
    user_df = pd.merge(user_df, df.groupby(['UserId','video_creator']).size().groupby(['UserId']).max().reset_index().rename(columns={0:'replay_creator_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_creator']).size().groupby(['UserId']).max().reset_index().rename(columns={0:'reshow_creator_max'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_creator']).size().groupby(['UserId']).max().reset_index().rename(columns={0: 'reclick_creator_max'}), on=['UserId'], how='left')

    user_df = pd.merge(user_df,df.groupby(['UserId', 'video_creator']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'replay_creator_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_creator']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reshow_creator_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_creator']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reclick_creator_mean'}), on=['UserId'], how='left')

    temp = df.groupby(['UserId', 'video_creator']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby('UserId').count().drop(['video_creator'], axis=1).rename(columns={'size': 'replay_creator_count'}), on=['UserId'], how='left')

    temp = df[df['show'] == 1].groupby(['UserId', 'video_creator']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby('UserId').count().drop(['video_creator'], axis=1).rename(columns={'size': 'reshow_creator_count'}), on=['UserId'], how='left')

    temp = df[df['click'] == 1].groupby(['UserId', 'video_creator']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df,temp[temp['size']> 1].groupby('UserId').count().drop(['video_creator'],axis=1).rename(columns={'size':'reclick_creator_count'}), on=['UserId'], how='left')

    #video_id
    user_df = pd.merge(user_df,df.groupby(['UserId','video_id']).size().groupby(['UserId']).max().reset_index().rename(columns ={0: 'replay_video_max'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_id']).size().groupby(['UserId']).max().reset_index().rename(columns={0:'reshow_video_max'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_id']).size().groupby(['UserId']).max().reset_index().rename(columns={0: 'reclick_video_max'}), on=['UserId'], how='left')

    user_df = pd.merge(user_df,df.groupby(['UserId', 'video_id']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'replay_video_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_id']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reshow_video_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_id']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reclick_video_mean'}), on=['UserId'], how='left')

    temp = df.groupby(['UserId', 'video_id']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df,temp[temp['size']>1].groupby(['UserId']).count().drop(['video_id'],axis=1).rename(columns={'size':'replay_video_count'}), on=['UserId'], how='left')

    temp = df[df['show'] == 1].groupby(['UserId', 'video_id']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby(['UserId']).count().drop(['video_id'], axis=1).rename(columns={'size': 'reshow_video_count'}), on=['UserId'], how='left')

    temp = df[df['click'] == 1].groupby(['UserId', 'video_id']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby(['UserId']).count().drop(['video_id'], axis=1).rename(columns={'size': 'reclick_video_count'}), on=['UserId'], how='left')

    #video_type
    user_df = pd.merge(user_df,df.groupby(['UserId', 'video_type']).size().groupby(['UserId']).max().reset_index().rename(columns={0: 'replay_type_max'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_type']).size().groupby(['UserId']).max().reset_index().rename(columns={0: 'reshow_type_max'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_type']).size().groupby(['UserId']).max().reset_index().rename(columns={0: 'reclick_type_max'}), on=['UserId'], how='left')

    user_df = pd.merge(user_df,df.groupby(['UserId', 'video_type']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'replay_type_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_type']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reshow_type_mean'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_type']).size().groupby(['UserId']).mean().reset_index().rename(columns={0: 'reclick_type_mean'}), on=['UserId'], how='left')

    temp = df.groupby(['UserId', 'video_type']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby(['UserId']).count().drop(['video_type'], axis=1).rename(columns={'size': 'replay_type_count'}), on=['UserId'], how='left')

    temp = df[df['show'] == 1].groupby(['UserId', 'video_type']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby(['UserId']).count().drop(['video_type'], axis=1).rename(columns={'size': 'reshow_type_count'}), on=['UserId'], how='left')

    temp = df[df['click'] == 1].groupby(['UserId', 'video_type']).size().reset_index().rename(columns={0: 'size'})
    user_df = pd.merge(user_df, temp[temp['size'] > 1].groupby(['UserId']).count().drop(['video_type'], axis=1).rename(columns={'size': 'reclick_type_count'}), on=['UserId'], how='left')
    del temp

    print('END get deep interest feat ', (time.time() - st), 's')
    return user_df


def get_deep_interval_feat(user_df,df):
    df['intervel'] = df['play_stamp'].diff()
    df['interval'][df['interval'] < 0] = np.nan

    df['interval_flag'] = df['interval'] / df['interval']
    df['interval_record'] = df['interval'].replace(0, np.nan)

    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].sum().reset_index().rename({'interval_record': 'interval_sum'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].mean().reset_index().rename({'interval_record': 'interval_mean'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].median().reset_index().rename({'interval_record': 'interval_median'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].max().reset_index().rename({'interval_record': 'interval_max'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].min().reset_index().rename({'interval_record': 'interval_min'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].std().reset_index().rename({'interval_record': 'interval_std'}, axis='columns'), on='UserId', how='left')
    user_df = pd.merge(user_df, df.groupby(by=['UserId'])['interval_record'].skew().reset_index().rename({'interval_record': 'interval_skew'}, axis='columns'), on='UserId', how='left')
    user_df['interval_diff'] = user_df['interval_max'] - user_df['interval_min']

    temp = df.groupby(by=['UserId'])['interval_flag'].sum().astype(int).reset_index().rename( {'interval_flag': 'interval_cnt'}, axis='columns')
    user_df = pd.merge(user_df, temp, on='UserId', how='left')
    del temp
    gc.collect()
    interval_cols = ['show_count_all','show_count', 'click_count', 'comment_count','like_count','share_count', 'video_id_sum',
                     'video_tag_sum','video_creator_sum','diff_time_play_count_x','diff_time_show_count_x','diff_time_click_count_x','diff_time_like_count_x',
                     'time_play_count','time_show_count', 'time_click_count','time_like_count','replay_creator_count','reshow_creator_count', 'reclick_creator_count',
                     'replay_video_count', 'reshow_video_count','reclick_video_count', 'replay_type_count','reshow_type_count',
                     'reclick_type_count','diff_time_play_count_y','diff_time_show_count_y', 'diff_time_click_count_y',
                     'diff_time_like_count_y', 'fav_play_type_count','fav_show_type_count', 'fav_click_type_count','fav_like_type_count','interval_sum']
    for col in interval_cols:
        user_df[col + '_interval_cnt_ratio'] = df[col] / df['interval_cnt']
        user_df[col + '_interval_sum_ratio'] = df[col] / df['interval_sum']
        user_df[col + '_interval_max_ratio'] = df[col] / df['interval_max']
        user_df[col + '_interval_min_ratio'] = df[col] / df['interval_min']
        user_df[col + '_interval_mean_ratio'] = df[col] / df['interval_mean']
        user_df[col + '_interval_median_ratio'] = df[col] / df['interval_median']
        user_df[col + '_interval_diff_ratio'] = df[col] / df['interval_diff']

    return user_df


def get_absence_feat(user_df,df):
    #读取文件后直接跑不用其他任何函数
    df[['gender','age','edu']] = df[['gender','age','edu']].replace('-', np.nan)
    user_df = pd.merge(user_df, df['gender'].groupby(df['UserId']).apply(lambda x: x.isnull().sum()).reset_index().rename(columns = {'gender':'gender'+'_nan_count'}),on=['UserId'],how='left')
    user_df = pd.merge(user_df, df['gender'].groupby(df['UserId']).apply(lambda x: x.isnull().sum()/x.size).reset_index().rename(columns = {'gender': 'gender' + '_nan_ratio'}),on=['UserId'],how='left')

    user_df = pd.merge(user_df, df['edu'].groupby(df['UserId']).apply(lambda x: x.isnull().sum()).reset_index().rename(columns={'edu': 'edu' + '_nan_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df['edu'].groupby(df['UserId']).apply(lambda x: x.isnull().sum() / x.size).reset_index().rename(columns={'edu': 'edu' + '_nan_ratio'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df['age'].groupby(df['UserId']).apply(lambda x: x.isnull().sum()).reset_index().rename(columns={'age': 'age' + '_nan_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df,df['age'].groupby(df['UserId']).apply(lambda x: x.isnull().sum() / x.size).reset_index().rename(columns={'age': 'age' + '_nan_ratio'}), on=['UserId'], how='left')
    return user_df
'''
def get_overoneday_feat(user_df,df):
    df['end_time'] = df['play_time']+df['play_time_length'].replace('-',0).astype(float)/3600
    df['is_overday'] = 0
    df['is_overday'][df['end_time']>24] =1
    pd.merge()
'''


def find_overoneday(df):
    time_stamp_play = list(df['play_stamp'] / 1000)
    localtime = []
    for i in time_stamp_play:
        localtime.append(time.localtime(i))
    df['play_time'] = [localtime[i].tm_hour + localtime[i].tm_min / 60 + localtime[i].tm_sec/3600 for i in range(len(localtime))]
    df['end_time'] = df['play_time'] + df['play_time_length'].replace('-', 0).astype(float) / 3600
    df['is_overday'] = 0
    df['is_overday'][df['end_time'] > 24] = 1

    return df['is_overday'].groupby(df['UserId']).max().reset_index()


def get_fav_video_type(user_df,df):
    #操作顺序 读完数据后 使用直接使用usage compression函数 然后运行本函数 我这一跑通
    #为了避免使用 decode函数 不使用get_user_df 函数
    #运行函数前如下建立user_df
    #user_df =pd.DataFrame()
    #user_df['UserId'] = df['UserId'].drop_duplicates()
    print("Start get fav video type")
    st = time.time()

    user_df = pd.merge(user_df, df.groupby(['UserId','video_type']).size().reset_index().groupby('UserId').max().rename(columns={'video_type':'fav_play_type',0:'fav_play_type_count'}), on=['UserId'], how='left')
    user_df = pd.merge(user_df, df.groupby(['UserId','video_type']).size().reset_index().groupby('UserId').apply(lambda x:x.max()[0]/x.sum()[0]).reset_index().rename(columns={0:'fav_play_type_ratio'}), on=['UserId'], how='left')

    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').max().rename(columns={'video_type': 'fav_show_type', 0: 'fav_show_type_count'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['show'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').apply(lambda x: x.max()[0] / x.sum()[0]).reset_index().rename(columns={0: 'fav_show_type_ratio'}), on=['UserId'],how='left')

    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').max().rename(columns={'video_type': 'fav_click_type', 0: 'fav_click_type_count'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['click'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').apply(lambda x: x.max()[0] / x.sum()[0]).reset_index().rename(columns={0: 'fav_click_type_ratio'}), on=['UserId'],how='left')

    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').max().rename(columns={'video_type': 'fav_like_type', 0: 'fav_like_type_count'}), on=['UserId'],how='left')
    user_df = pd.merge(user_df, df[df['like'] == 1].groupby(['UserId', 'video_type']).size().reset_index().groupby('UserId').apply(lambda x: x.max()[0] / x.sum()[0]).reset_index().rename(columns={0: 'fav_like_type_ratio'}), on=['UserId'],how='left')

    print('END get fav video type ', (time.time() - st), 's')
    return user_df


