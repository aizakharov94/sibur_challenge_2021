
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import dill

MODEL_FILE = pathlib.Path(__file__).parent.joinpath("model.pkl")
MODEL_CALSS_0_FILE = pathlib.Path(__file__).parent.joinpath("model_class_0.pkl")
OHE_FILE = pathlib.Path(__file__).parent.joinpath("ohe.pkl")


prev_cols_len = 360
predictions_len = 30
key_cols = ["material_code", "company_code", "country", "region", "manager_code"]
cat_cols = ['material_lvl1_name', 'material_lvl2_name', 'material_lvl3_name']

prev_cols = ["prev_" + str(i) for i in range(1, prev_cols_len + 1)][::-1]
prev_cols_days = ["prev_" + str(i) + '_day' for i in range(1, 31)][::-1]

holidays_list = ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-08', '2018-02-23',
                 '2018-03-08', '2018-03-09', '2018-04-30', '2018-05-01', '2018-05-02', '2018-05-09', '2018-06-11',
                 '2018-06-12', '2018-11-05', '2018-12-31', '2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                 '2019-01-07', '2019-01-08', '2019-03-08', '2019-05-01', '2019-05-02', '2019-05-03', '2019-05-09',
                 '2019-05-10', '2019-06-12', '2019-11-04', '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-03',
                 '2020-01-06', '2020-01-07', '2020-01-08', '2020-02-24', '2020-03-09', '2020-05-01', '2020-05-04',
                 '2020-05-05', '2020-05-11', '2020-06-12', '2020-11-04', '2020-12-31', '2021-01-01', '2021-01-04',
                 '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-02-22', '2021-02-23', '2021-03-08',
                 '2021-05-03', '2021-05-04', '2021-05-05', '2021-05-10', '2021-06-14', '2021-11-04', '2021-11-05',
                 '2021-12-31', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-02-23',
                 '2022-03-07', '2022-03-08', '2022-05-02', '2022-05-03', '2022-05-09', '2022-05-10', '2022-06-13',
                 '2022-11-04']

def get_main_df(table):
    
    # Download table
    df = table.copy()
    df = df.groupby(list(df.columns)[:-1])['volume'].sum().reset_index()
    df.rename({'volume': 'target'}, axis=1, inplace=True)

    # All unique points
    unique_rows = df[key_cols].drop_duplicates()
    unique_rows['key'] = 0

    # Dates range
    dates_data = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max())})
    dates_data['key'] = 0
    all_rows_date = unique_rows.merge(dates_data, how='outer', on=['key']).drop(['key'], axis=1)

    # Min dates
    min_dates = df.groupby(key_cols)['date'].min().reset_index()
    min_dates.rename({'date': 'min_date'}, axis=1, inplace=True)
    all_rows_date = all_rows_date.merge(min_dates, how='left', on=key_cols)
    all_rows_date = all_rows_date[all_rows_date['date'] >= all_rows_date['min_date']].reset_index(drop=True)
    all_rows_date.drop(['min_date'], axis=1, inplace=True)

    # Join main DataFrame
    df = all_rows_date.merge(df, how='left', on=key_cols + ['date'])
    df['target'] = df['target'].fillna(0).astype(int)
    
    df = df.sort_values(key_cols + ['date']).reset_index(drop=True)
    df.material_lvl1_name = df.material_lvl1_name.fillna(method='ffill')
    df.material_lvl2_name = df.material_lvl2_name.fillna(method='ffill')
    df.material_lvl3_name = df.material_lvl3_name.fillna(method='ffill')
    df.contract_type = df.contract_type.fillna(method='ffill')
    df.month = pd.to_datetime(df['date'].astype(str).str[:-2] + '01')
    
    return df

def get_test_data(df, start_date):    
    df_test = df[key_cols + ['material_lvl1_name', 'material_lvl2_name', 'material_lvl3_name']]\
                                                                        .drop_duplicates().reset_index(drop=True)
    df_test = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    df_test['date'] = start_date
    df_test['month'] = start_date
    df_test['target'] = -1
    df = pd.concat([df, df_test]).reset_index(drop=True)
    df = df.sort_values(key_cols + ['date']).reset_index(drop=True)
    
    df = df[df.date > start_date - pd.Timedelta(days=prev_cols_len + 45)].reset_index(drop=True)
    for i in range(1, prev_cols_len + 1):
        df['prev_' + str(i)] = df.groupby(key_cols)["target"].shift(i).rolling(30).sum()
    
    df = df[df.date > start_date - pd.Timedelta(days=45)].reset_index(drop=True)
    for i in range(1, 31):
        df['prev_' + str(i) + '_day'] = df.groupby(key_cols)["target"].shift(i).rolling(1).sum()
    
    df = df[df.date == start_date].reset_index(drop=True)
    return df

def get_dayoff_features():    
    holidays_table = pd.DataFrame({'date': holidays_list})
    holidays_table['year'] = holidays_table['date'].str[:4].astype(int)
    holidays_table = holidays_table[['year', 'date']]
    dates = holidays_table['date'].astype(str)    
    data_hol = pd.DataFrame(pd.date_range('2018-01-01', '2023-01-01'), columns=['merge_date'])
    data_hol['merge_date'] = data_hol['merge_date'].astype(str)
    data_hol = data_hol.merge(pd.DataFrame({'merge_date': dates, 'is_holiday': 1}), how='left', on=['merge_date'])
    data_hol['is_holiday'] = data_hol['is_holiday'].fillna(0).astype(int)
    data_hol['weekday'] = pd.to_datetime(data_hol['merge_date']).dt.dayofweek.astype(int)
    data_hol['is_weekend'] = data_hol['weekday'].isin({5,6}).astype(int)
    data_hol['is_dayoff'] = (data_hol['is_holiday'] | data_hol['is_weekend']).astype(int)
    data_hol['merge_date'] = pd.to_datetime(data_hol['merge_date'])
    
    data_hol['key'] = 0
    data_hol = data_hol[['merge_date', 'key']].merge(
                data_hol.rename({'merge_date': 'dt'}, axis=1), how='outer', on='key')
    data_hol['merge_date_end'] = data_hol['merge_date'] + pd.DateOffset(months=1)
    data_hol = data_hol[(data_hol.merge_date <= data_hol['dt']) &\
                        (data_hol.merge_date_end > data_hol['dt'])].reset_index(drop=True)
    data_hol = data_hol.groupby(['merge_date']).agg({'is_holiday': ['sum', 'mean'],
                                                     'is_weekend': ['sum', 'mean'],
                                                     'is_dayoff': ['sum', 'mean']})
    data_hol.columns = ["_".join(x) for x in data_hol.columns.ravel()]
    data_hol = data_hol.reset_index().rename({'merge_date': 'date'}, axis=1)
    return data_hol

def get_cat_features_transform(test, ohe):
    test['is_contract'] = test['contract_type'].isin(['Контракт', 'Contract + Spot']) * 1
    test['is_spot'] = test['contract_type'].isin(['Спот', 'Contract + Spot']) * 1
    test.drop('contract_type', axis=1, inplace=True)
    for col in cat_cols:
        test[col] = test[col].str.replace(' ', '_')
    for col in cat_cols:
        test['for_cat_' + col] = test[col].copy()
    test = ohe.transform(test)
    return test

def get_stat_features(dfs):
    
    ### Statistics from previous series values
    # 1 month
    dfs['1_month_median'] = dfs[prev_cols[-30:]].median(1)
    dfs['1_month_mean'] = dfs[prev_cols[-30:]].mean(1)
    dfs['1_month_std'] = dfs[prev_cols[-30:]].std(1)
    dfs['1_month_max'] = dfs[prev_cols[-30:]].max(1)
    dfs['1_month_min'] = dfs[prev_cols[-30:]].min(1)
    dfs['1_month_zeros_prop'] = (dfs[prev_cols[-30:]] == 0).mean(1)

    # 2 month
    dfs['2_month_median'] = dfs[prev_cols[-60:-30]].median(1)
    dfs['2_month_mean'] = dfs[prev_cols[-60:-30]].mean(1)
    dfs['2_month_std'] = dfs[prev_cols[-60:-30]].std(1)
    dfs['2_month_max'] = dfs[prev_cols[-60:-30]].max(1)
    dfs['2_month_min'] = dfs[prev_cols[-60:-30]].min(1)
    dfs['2_month_zeros_prop'] = (dfs[prev_cols[-60:-30]] == 0).mean(1)

    # 1_2 month changes
    dfs['month_median_change'] = (dfs['1_month_median'] - dfs['2_month_median']) / (dfs['2_month_median'] + 1e-6)
    dfs['month_mean_change'] = (dfs['1_month_mean'] - dfs['2_month_mean']) / (dfs['2_month_mean'] + 1e-6)
    dfs['month_std_change'] = (dfs['1_month_std'] - dfs['2_month_std']) / (dfs['2_month_std'] + 1e-6)
    dfs['month_max_change'] = (dfs['1_month_max'] - dfs['2_month_max']) / (dfs['2_month_max'] + 1e-6)
    dfs['month_min_change'] = (dfs['2_month_min'] - dfs['2_month_min']) / (dfs['2_month_min'] + 1e-6)
    dfs['month_zeros_prop_change'] = (dfs['1_month_zeros_prop'] - dfs['2_month_zeros_prop']) /\
                                                                        (dfs['2_month_zeros_prop'] + 1e-6)
    # 1-3 months
    dfs['1_3_month_median'] = dfs[prev_cols[-90:]].median(1)
    dfs['1_3_month_mean'] = dfs[prev_cols[-90:]].mean(1)
    dfs['1_3_month_std'] = dfs[prev_cols[-90:]].std(1)
    dfs['1_3_month_max'] = dfs[prev_cols[-90:]].max(1)
    dfs['1_3_month_min'] = dfs[prev_cols[-90:]].min(1)
    dfs['1_3_month_zeros_prop'] = (dfs[prev_cols[-90:]] == 0).mean(1)
    
    # 1-3 changes to 1 month
    dfs['glob_month_median_change'] = (dfs['1_month_median'] - dfs['1_3_month_median']) /\
                                                                (dfs['1_3_month_median'] + 1e-6)
    dfs['glob_month_mean_change'] = (dfs['1_month_mean'] - dfs['1_3_month_mean']) /\
                                                                (dfs['1_3_month_mean'] + 1e-6)
    dfs['glob_month_max_change'] = (dfs['1_month_max'] - dfs['1_3_month_max']) / (dfs['1_3_month_max'] + 1e-6)
    dfs['glob_month_min_change'] = (dfs['1_month_min'] - dfs['1_3_month_min']) / (dfs['1_3_month_min'] + 1e-6)
    dfs['glob_month_zeros_prop_change'] = (dfs['1_month_zeros_prop'] - dfs['1_3_month_zeros_prop']) /\
                                                                        (dfs['1_3_month_zeros_prop'] + 1e-6)
    
    # year
    dfs['year_median'] = dfs[prev_cols[-360:]].median(1)
    dfs['year_mean'] = dfs[prev_cols[-360:]].mean(1)
    dfs['year_std'] = dfs[prev_cols[-360:]].std(1)
    dfs['year_max'] = dfs[prev_cols[-360:]].max(1)
    dfs['year_min'] = dfs[prev_cols[-360:]].min(1)
    dfs['year_zeros_prop'] = (dfs[prev_cols[-360:]] == 0).mean(1)
    
    # half year 1
    dfs['1_half_year_median'] = dfs[prev_cols[-180:]].median(1)
    dfs['1_half_year_mean'] = dfs[prev_cols[-180:]].mean(1)
    dfs['1_half_year_std'] = dfs[prev_cols[-180:]].std(1)
    dfs['1_half_year_max'] = dfs[prev_cols[-180:]].max(1)
    dfs['1_half_year_min'] = dfs[prev_cols[-180:]].min(1)
    dfs['1_half_year_zeros_prop'] = (dfs[prev_cols[-180:]] == 0).mean(1)
    
    # half year 2
    dfs['2_half_year_median'] = dfs[prev_cols[-360:-180]].median(1)
    dfs['2_half_year_mean'] = dfs[prev_cols[-360:-180]].mean(1)
    dfs['2_half_year_std'] = dfs[prev_cols[-360:-180]].std(1)
    dfs['2_half_year_max'] = dfs[prev_cols[-360:-180]].max(1)
    dfs['2_half_year_min'] = dfs[prev_cols[-360:-180]].min(1)
    dfs['2_half_year_zeros_prop'] = (dfs[prev_cols[-360:-180]] == 0).mean(1)
    
    # half year 1 2 stats
    dfs['half_year_median_change'] = (dfs['1_half_year_median'] - dfs['2_half_year_median']) /\
                                                                            (dfs['2_half_year_median'] + 1e-6)
    dfs['half_year_week_mean_change'] = (dfs['1_half_year_mean'] - dfs['2_half_year_mean']) /\
                                                                            (dfs['2_half_year_mean'] + 1e-6)
    dfs['half_year_week_std_change'] = (dfs['1_half_year_std'] - dfs['2_half_year_std']) /\
                                                                            (dfs['2_half_year_std'] + 1e-6)
    dfs['half_year_week_max_change'] = (dfs['1_half_year_max'] - dfs['2_half_year_max']) /\
                                                                            (dfs['2_half_year_max'] + 1e-6)
    dfs['half_year_week_min_change'] = (dfs['1_half_year_zeros_prop'] - dfs['2_half_year_min']) /\
                                                                            (dfs['2_half_year_min'] + 1e-6)
    dfs['half_year_week_zeros_prop_change'] = (dfs['1_half_year_zeros_prop'] - dfs['2_half_year_zeros_prop']) /\
                                                                            (dfs['2_half_year_zeros_prop'] + 1e-6)
    
    # year changes to 1 month
    dfs['glob_year_median_change'] = (dfs['1_month_median'] - dfs['year_median']) / (dfs['year_median'] + 1e-6)
    dfs['glob_year_mean_change'] = (dfs['1_month_mean'] - dfs['year_mean']) / (dfs['1_3_month_mean'] + 1e-6)
    dfs['glob_year_max_change'] = (dfs['1_month_max'] - dfs['year_max']) / (dfs['year_max'] + 1e-6)
    dfs['glob_year_min_change'] = (dfs['1_month_min'] - dfs['year_min']) / (dfs['year_min'] + 1e-6)
    dfs['glob_year_zeros_prop_change'] = (dfs['1_month_zeros_prop'] - dfs['year_zeros_prop']) /\
                                                                        (dfs['year_zeros_prop'] + 1e-6) 
    # half year changes to 1 month
    dfs['glob_half_year_median_change'] = (dfs['1_month_median'] - dfs['1_half_year_median']) /\
                                                                    (dfs['1_half_year_median'] + 1e-6)
    dfs['glob_half_year_mean_change'] = (dfs['1_month_mean'] - dfs['1_half_year_mean']) /\
                                                                    (dfs['1_half_year_mean'] + 1e-6)
    dfs['glob_half_year_max_change'] = (dfs['1_month_max'] - dfs['1_half_year_max']) /\
                                                                    (dfs['1_half_year_max'] + 1e-6)
    dfs['glob_half_year_min_change'] = (dfs['1_month_min'] - dfs['1_half_year_min']) /\
                                                                    (dfs['1_half_year_min'] + 1e-6)
    dfs['glob_half_year_zeros_prop_change'] = (dfs['1_month_zeros_prop'] - dfs['1_half_year_zeros_prop']) /\
                                                                        (dfs['1_half_year_zeros_prop'] + 1e-6) 
    
    # 1 week
    dfs['1_week_median'] = dfs[prev_cols[-7:]].median(1)
    dfs['1_week_mean'] = dfs[prev_cols[-7:]].mean(1)
    dfs['1_week_std'] = dfs[prev_cols[-7:]].std(1)
    dfs['1_week_max'] = dfs[prev_cols[-7:]].max(1)
    dfs['1_week_min'] = dfs[prev_cols[-7:]].min(1)
    dfs['1_week_zeros_prop'] = (dfs[prev_cols[-7:]] == 0).mean(1)

    # 2 week
    dfs['2_week_median'] = dfs[prev_cols[-14:-7]].median(1)
    dfs['2_week_mean'] = dfs[prev_cols[-14:-7]].mean(1)
    dfs['2_week_std'] = dfs[prev_cols[-14:-7]].std(1)
    dfs['2_week_max'] = dfs[prev_cols[-14:-7]].max(1)
    dfs['2_week_min'] = dfs[prev_cols[-14:-7]].min(1)
    dfs['2_week_zeros_prop'] = (dfs[prev_cols[-14:-7]] == 0).mean(1)
    
    # 1_2 week changes
    dfs['week_median_change'] = (dfs['1_week_median'] - dfs['2_week_median']) / (dfs['2_week_median'] + 1e-6)
    dfs['week_mean_change'] = (dfs['1_week_mean'] - dfs['2_week_mean']) / (dfs['2_week_mean'] + 1e-6)
    dfs['week_std_change'] = (dfs['1_week_std'] - dfs['2_week_std']) / (dfs['2_week_std'] + 1e-6)
    dfs['week_max_change'] = (dfs['1_week_max'] - dfs['2_week_max']) / (dfs['2_week_max'] + 1e-6)
    dfs['week_min_change'] = (dfs['2_week_min'] - dfs['2_week_min']) / (dfs['2_week_min'] + 1e-6)
    dfs['week_zeros_prop_change'] = (dfs['1_week_zeros_prop'] - dfs['2_week_zeros_prop']) /\
                                                                        (dfs['2_week_zeros_prop'] + 1e-6)
    # 1 day statistics
    dfs['1_2_day_perc'] = (dfs['prev_1'] - dfs['prev_2']) / (dfs['prev_2'] + 1e-6)
    dfs['1_2_day_diff'] = dfs['prev_1'] - dfs['prev_2']
    dfs['1_day_month_change_mean'] = (dfs['prev_1'] - dfs['1_month_mean']) / (dfs['1_month_mean'] + 1e-6)
    dfs['1_day_month_change_median'] = (dfs['prev_1'] - dfs['1_month_median']) / (dfs['1_month_median'] + 1e-6)
    dfs['1_day_month_change_max'] = (dfs['prev_1'] - dfs['1_month_max']) / (dfs['1_month_max'] + 1e-6)
    dfs['1_day_month_change_min'] = (dfs['prev_1'] - dfs['1_month_min']) / (dfs['1_month_min'] + 1e-6)
    dfs['1_day_week_change_mean'] = (dfs['prev_1'] - dfs['1_week_mean']) / (dfs['1_week_mean'] + 1e-6)
    dfs['1_day_week_change_median'] = (dfs['prev_1'] - dfs['1_week_median']) / (dfs['1_week_median'] + 1e-6)
    dfs['1_day_week_change_max'] = (dfs['prev_1'] - dfs['1_week_max']) / (dfs['1_week_max'] + 1e-6)
    dfs['1_day_week_change_min'] = (dfs['prev_1'] - dfs['1_week_min']) / (dfs['1_week_min'] + 1e-6)  
    
    return dfs

def get_last_action(df):
    dense = ((df[prev_cols] != 0) & (~df[prev_cols].isnull())).multiply(range(prev_cols_len, 0, -1), axis=1)
    dense[dense == 0] = 999
    df['days_from_last_non_zero_action'] = dense.idxmin(axis=1).str.split('_').str[-1].astype(int)
    df['last_non_zero_value'] = np.array(df[prev_cols])[range(df.shape[0]),
                                                prev_cols_len - df['days_from_last_non_zero_action'].values]
    dense = ((df[prev_cols] == 0) & (~df[prev_cols].isnull())).multiply(range(prev_cols_len, 0, -1), axis=1)
    dense[dense == 0] = 999
    df['days_from_last_zero_action'] = dense.idxmin(axis=1).str.split('_').str[-1].astype(int)
    return df

def add_lr_features(df):   
    buffs = pd.DataFrame((df[prev_cols[1:]].values != df[prev_cols[:-1]].values) |\
                         (df[prev_cols[1:]].values == 0)).multiply(range(prev_cols_len - 1, 0, -1), axis=1)
    buffs = (buffs != 0).multiply(range(prev_cols_len - 1), axis=1)
    df['equal_days_befor_target'] = prev_cols_len - 1 - buffs.idxmax(1)
    return df

def get_zeros_mass_features(df):
    def get_zeros_length(x):
        return np.arange(len(x))[(x == 0) & (np.concatenate([x[1:], np.array([-1])]) != 0)] -\
               np.arange(len(x))[(x == 0) & (np.concatenate([np.array([-1]), x[:-1]]) != 0)] + 1
    df['zeros_masses'] = df[prev_cols].apply(lambda x: get_zeros_length(x.values), axis=1)
    df['last_zeros_len'] = df['zeros_masses'].apply(lambda x: x[-1] if len(x) > 0 else -1)
    df['zeros_mass_mean'] = df['zeros_masses'].apply(lambda x: np.mean(x) if len(x) > 0 else -1)
    df['zeros_mass_median'] = df['zeros_masses'].apply(lambda x: np.median(x) if len(x) > 0 else -1)
    df['zeros_mass_std'] = df['zeros_masses'].apply(lambda x: np.std(x) if len(x) > 0 else -1)
    df['zeros_mass_cnt'] = df['zeros_masses'].apply(lambda x: len(x))
    df['zeros_mass_max'] = df['zeros_masses'].apply(lambda x: np.max(x) if len(x) > 0 else -1)
    df.drop(['zeros_masses'], axis=1, inplace=True)
    return df

def get_last_days_features(df):
    df['last_day_total'] = df['prev_1_day'] * 30
    df['last_week_total'] = df[prev_cols_days[-7:]].sum(1) / 7 * 30
    df['last_2_week_total'] = df[prev_cols_days[-14:-7]].sum(1) / 7 * 30
    df['last_month_total'] = df['prev_1'].copy()
    
    df['perc_last_day_week_total'] = df['last_day_total'] / (df['last_week_total'] + 1e-6)
    df['perc_last_day_month_total'] = df['last_day_total'] / (df['last_month_total'] + 1e-6)
    df['perc_last_week_month_total'] = df['last_week_total'] / (df['last_month_total'] + 1e-6)
    df['perc_last_double_weeks_month_total'] = (df['last_week_total'] + df['last_week_total']) /\
                                                                        (2 * df['last_month_total'] + 1e-6)
    df['perc_last_week_2_week_total'] = df['last_week_total'] / (df['last_2_week_total'] + 1e-6)
    df['perc_last_week_double_week_total'] = df['last_week_total'] * 2 /\
                                                (df['last_2_week_total'] + df['last_week_total'] + 1e-6)
    return df

def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    test = get_main_df(df)
    test = get_test_data(test, month)
    ohe = pickle.load(open(OHE_FILE, 'rb'))
    test = get_cat_features_transform(test, ohe)
    test = test.merge(get_dayoff_features(), how='left', on=['date'])
    test = get_stat_features(test)
    test = get_last_action(test)
    test = add_lr_features(test)
    test = get_zeros_mass_features(test)
    test = get_last_days_features(test)
    
    model_regression = dill.load(open(MODEL_FILE, "rb"))
    predictions_regression = model_regression.predict(test)
    
    model_classification_0 = pickle.load(open(MODEL_CALSS_0_FILE, 'rb'))
    predictions_classification_0 = model_classification_0.predict(test[model_classification_0.feature_name()])

    preds_df = test[key_cols + ['prev_1']].copy()
    preds_df["preds"] = np.expm1(predictions_regression)
    preds_df["pred_surge_0_first"] = predictions_classification_0
    
    preds_df['prediction'] = preds_df['preds'] * (preds_df['prev_1'] != 0) +\
          preds_df['preds'] * (preds_df['prev_1'] == 0) * (preds_df['pred_surge_0_first'] > 0.2) * 1.04 +\
          preds_df['preds'] * (preds_df['prev_1'] == 0) * (preds_df['pred_surge_0_first'] <= 0.2) * 0.50

    preds_df['prediction'] = np.clip(preds_df['prediction'], 0, None)
    
    return preds_df[key_cols + ['prediction']]
