import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import os
import warnings; warnings.filterwarnings('ignore')
SEED = 42; np.random.seed(SEED)
sales = pd.read_csv('sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
sub_filename = 'sample_submission.csv' if os.path.exists('sample_submission.csv') else 'submission.csv'
raw_sub = pd.read_csv(sub_filename)
test_dates = pd.to_datetime(raw_sub['Date'], errors='coerce')

hols = {(1,1),(4,30),(5,1),(9,2),(11,11),(12,12),(12,24),(12,25)}
tet = {2012:'2012-01-23',2013:'2013-02-10',2014:'2014-01-31',2015:'2015-02-19',
       2016:'2016-02-08',2017:'2017-01-28',2018:'2018-02-16',2019:'2019-02-05',
       2020:'2020-01-25',2021:'2021-02-12',2022:'2022-02-01',2023:'2023-01-22',2024:'2024-02-10'}
tet = {k:pd.Timestamp(v) for k,v in tet.items()}

ghost_months = [
    ('2012-08-17', '2012-09-15'), ('2013-08-07', '2013-09-04'),
    ('2014-07-27', '2014-08-24'), ('2015-08-14', '2015-09-12'),
    ('2016-08-03', '2016-08-31'), ('2017-08-22', '2017-09-19'),
    ('2018-08-11', '2018-09-09'), ('2019-08-01', '2019-08-29'),
    ('2020-08-19', '2020-09-16'), ('2021-08-08', '2021-09-06'),
    ('2022-07-29', '2022-08-26'), ('2023-08-16', '2023-09-14'),
    ('2024-08-04', '2024-09-02')
]

def make_synthesis_features(df):
    df = df.copy()
    df['Month']           = df['Date'].dt.month
    df['DayOfWeek']       = df['Date'].dt.dayofweek
    df['IsWeekend']       = (df['DayOfWeek'] >= 5).astype(int)
    df['DayOfYear']       = df['Date'].dt.dayofyear
    df['Day']             = df['Date'].dt.day
    df['DaysInMonth']     = df['Date'].dt.days_in_month
    
    df['IsHoliday']       = df['Date'].apply(lambda d: int((d.month,d.day) in hols))
    df['TetDays']         = df['Date'].apply(lambda d: int((d-tet[d.year]).days) if d.year in tet else 0)
    df['TetPre7']         = df['TetDays'].between(-7,-1).astype(int)
    df['TetPost7']        = df['TetDays'].between(0,7).astype(int)
    df['IsTetCrash']      = df['TetDays'].isin([0, 1, 2, 3]).astype(int)
    
    # Tháng cô hồn
    df['IsGhostMonth'] = 0
    for start, end in ghost_months:
        mask = (df['Date'] >= start) & (df['Date'] <= end)
        df.loc[mask, 'IsGhostMonth'] = 1
        
    is_double_day = (df['Month'] == df['Day'])
    df['IsMegaSale'] = (is_double_day & df['Month'].isin([11, 12])).astype(int)
    df['IsMidSale']  = (is_double_day & df['Month'].isin([9, 10])).astype(int)
    
    df['IsShopee15'] = (df['Day'] == 15).astype(int) 
    df['IsShopee25'] = (df['Day'] == 25).astype(int) 
    df['IsMegaShopee15'] = (df['IsShopee15'] & df['Month'].isin([11, 12])).astype(int)
    df['IsMegaShopee25'] = (df['IsShopee25'] & df['Month'].isin([11, 12])).astype(int)
    
    df['IsValentine'] = ((df['Month'] == 2) & (df['Day'] == 14)).astype(int)
    df['IsWomensDay'] = (((df['Month'] == 3) & (df['Day'] == 8)) | ((df['Month'] == 10) & (df['Day'] == 20))).astype(int)
    df['IsBlackFriday'] = ((df['Month'] == 11) & (df['DayOfWeek'] == 4) & (df['Day'] >= 23) & (df['Day'] <= 29)).astype(int)
    
    days_in_year = df['Date'].dt.is_leap_year.map({True: 366, False: 365})
    for k in [1,2,3,4,5]:
        df[f'sin_y_{k}'] = np.sin(2 * np.pi * k * df['DayOfYear'] / days_in_year)
        df[f'cos_y_{k}'] = np.cos(2 * np.pi * k * df['DayOfYear'] / days_in_year)
        
    for k in [1,2]:
        df[f'sin_m_{k}'] = np.sin(2 * np.pi * k * df['Day'] / df['DaysInMonth'])
        df[f'cos_m_{k}'] = np.cos(2 * np.pi * k * df['Day'] / df['DaysInMonth'])
        
    decay_1 = np.where(df['Day'] >= 1, np.exp(-(df['Day'] - 1) * 0.4), 0)
    decay_15 = np.where(df['Day'] >= 15, np.exp(-(df['Day'] - 15) * 0.4), 0)
    decay_25 = np.where(df['Day'] >= 25, np.exp(-(df['Day'] - 25) * 0.4), 0)
    df['SalaryBoost'] = np.maximum.reduce([decay_1, decay_15, decay_25]) 
    
    return df

train_df = make_synthesis_features(sales)
test_df = make_synthesis_features(pd.DataFrame({'Date': test_dates}))

cat_cols = ['Month', 'DayOfWeek', 'IsWeekend', 'IsGhostMonth']
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

FEATS = [c for c in train_df.columns if c not in ['Date','Revenue','COGS','TetDays', 'DaysInMonth']]
X_tr = train_df[FEATS]; X_te = test_df[FEATS]

y_rev_raw = train_df['Revenue']
y_cogs_raw = train_df['COGS']
y_rev_log = np.log1p(train_df['Revenue'])
y_cogs_log = np.log1p(train_df['COGS'])


CAT_N, LGB_N, LR = 2500, 2500, 0.008

cat_r = CatBoostRegressor(n_estimators=CAT_N, learning_rate=LR, depth=6, random_seed=SEED, verbose=0, loss_function='MAE', cat_features=cat_cols)
cat_c = CatBoostRegressor(n_estimators=CAT_N, learning_rate=LR, depth=6, random_seed=SEED, verbose=0, loss_function='MAE', cat_features=cat_cols)
cat_r.fit(X_tr, y_rev_raw); cat_c.fit(X_tr, y_cogs_raw)

lgb_r = LGBMRegressor(n_estimators=LGB_N, learning_rate=LR, num_leaves=31, random_state=SEED, verbose=-1, objective='mae')
lgb_c = LGBMRegressor(n_estimators=LGB_N, learning_rate=LR, num_leaves=31, random_state=SEED, verbose=-1, objective='mae')
lgb_r.fit(X_tr, y_rev_raw); lgb_c.fit(X_tr, y_cogs_raw)

pure_mae_rev = (0.60 * cat_r.predict(X_te)) + (0.40 * lgb_r.predict(X_te))
pure_mae_cogs = (0.60 * cat_c.predict(X_te)) + (0.40 * lgb_c.predict(X_te))

cat_log_r = CatBoostRegressor(n_estimators=CAT_N, learning_rate=LR, depth=6, random_seed=SEED, verbose=0, loss_function='RMSE', cat_features=cat_cols)
cat_log_c = CatBoostRegressor(n_estimators=CAT_N, learning_rate=LR, depth=6, random_seed=SEED, verbose=0, loss_function='RMSE', cat_features=cat_cols)
cat_log_r.fit(X_tr, y_rev_log); cat_log_c.fit(X_tr, y_cogs_log)

pred_log_rev = np.expm1(cat_log_r.predict(X_te))
pred_log_cogs = np.expm1(cat_log_c.predict(X_te))

final_rev = (0.85 * pure_mae_rev) + (0.15 * pred_log_rev)
final_cogs = (0.85 * pure_mae_cogs) + (0.15 * pred_log_cogs)

sub = raw_sub.copy()
sub['Revenue'] = np.clip(final_rev, a_min=0, a_max=None).round(2)

# VŨ KHÍ GỌT COGS CHUẨN V64 (Đã tạo ra 663k): KHÓA TRẦN 95%, KHÓA ĐÁY 70%
sub['COGS'] = np.clip(final_cogs, a_min=sub['Revenue'] * 0.70, a_max=sub['Revenue'] * 0.95).round(2)

sub['Date'] = pd.to_datetime(sub['Date'], errors='ignore').dt.strftime('%Y-%m-%d')

filename = 'The_Gridbreaker_Ultimate_Synthesis_V66.csv'
sub.to_csv(filename, index=False)
print("XONG")
import shap
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

explainer = shap.TreeExplainer(cat_r)
shap_values = explainer.shap_values(X_tr)

business_features = [
    'IsMegaSale', 'IsGhostMonth', 'SalaryBoost', 'IsShopee15', 
    'IsShopee25', 'IsWeekend', 'IsValentine', 'IsWomensDay', 'IsBlackFriday'
]

valid_biz_features = [f for f in business_features if f in X_tr.columns]
feature_indices = [X_tr.columns.get_loc(f) for f in valid_biz_features]

shap_values_biz = shap_values[:, feature_indices]
X_tr_biz = X_tr[valid_biz_features]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_biz, X_tr_biz, show=False)

plt.title("Mức độ tác động của các sự kiện kinh doanh và \n yếu tố văn hóa lên doanh thu dự báo", fontsize=15, weight='bold', pad=20)
plt.xlabel("Tác động tới Doanh Thu (VND)", fontsize=12)
plt.tight_layout()

plt.savefig('shap_executive_view.png', dpi=300, bbox_inches='tight')
print("XONG")
plt.show()