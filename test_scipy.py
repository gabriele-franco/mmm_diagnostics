from scipy.optimize import minimize
from main import values_model
import pandas as pd

def robyn_model_obj(summary_dict):
    result, model, summary_dict, df_saturation_ridge, lambda_value, Nrmse = robyn_model(df, all_media, summary_dict, window_start, window_end, date_var, df_prophet, all_vars, dep_var, df_window, use_intercept, prophet_future_dataframe_periods=14, prophet_seasonality_mode='additive')
    print(f'nrmse {nrmse}')
    return Nrmse

json_model=pd.read_json('RobynModel-7_946_5.json')
df=pd.read_csv('raw_data.csv')


df_holidays=pd.read_csv('dataset_holidays.txt', delimiter=',')


date_var,dep_var,window_start,window_end, all_media,all_vars,df_prophet, summary_dict,df_saturation_ridge,lambda_value,ridge_intercept,ridge_coefs,df_window,use_intercept,prophet_vars,prophet_country, prophet_freq=values_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive')


optimization_result = minimize(robyn_model_obj, **summary_dict)