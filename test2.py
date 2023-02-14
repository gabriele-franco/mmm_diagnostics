from func import values_model, prophet, ridge, rssd,show_nrmse,transformations_parameters
import numpy as np
import nevergrad as ng
import pandas as pd

json_model=pd.read_json('RobynModel-7_946_5.json')
df=pd.read_csv('raw_data_rf1.csv')


df_holidays=pd.read_csv('dataset_holidays.txt', delimiter=',')

def create_parametrization(summary_dict):
    parameters = {}
    for key, value in summary_dict.items():
        for i in value:
            if i =='coef':
                if summary_dict[key][i] != 0:
                    parameters[f'{key}_{i}']=ng.p.Scalar(lower=summary_dict[key][i]*0.5, upper=summary_dict[key][i]*1.5)
                else:
                    parameters[f'{key}_{i}']=ng.p.Scalar(lower=0, upper=20)
            else:
                if summary_dict[key][i] != 0:
                    parameters[f'{key}_{i}']=ng.p.Scalar(lower=summary_dict[key][i]*0.8, upper=summary_dict[key][i]*1.2)
                else:
                    parameters[f'{key}_{i}']=ng.p.Scalar(lower=0, upper=20)
    return parameters

date_var,dep_var,window_start,window_end, all_media,all_vars,df_prophet, summary_dict,lambda_value,df_window,use_intercept,prophet_vars,prophet_country, prophet_freq=values_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive')

def refresh_model(df,summary_dict,lambda_value, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
                national_holidays_abbreviation=prophet_country, future_dataframe_periods=14, freq=prophet_freq, seasonality_mode='additive'):

    def model_refreshed(**parameters):
        #create dataset with seasonality
        df_prophet = prophet(df, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
                national_holidays_abbreviation=prophet_country, future_dataframe_periods=future_dataframe_periods, freq=prophet_freq, seasonality_mode=seasonality_mode)

        #transform media variables into dim return and adstock
        df_saturation_ridge,ridge_intercept,ridge_coefs=transformations_parameters(df, all_media, parameters, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept)
        
        #add the coefficients into the function
        result, model=ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value, size=0.2, positive=True, random_state=42, coeffs=ridge_coefs[ : -1], intercept=ridge_intercept)
        
        nrmse=show_nrmse(result[dep_var], result['prediction'])
        
        decomp_rssd=rssd(df, ridge_coefs, verbose=False)
        
        return result, model, summary_dict,df_saturation_ridge,lambda_value,nrmse, decomp_rssd
    

    def objective_func(**parameters):
                #create dataset with seasonality
        df_prophet = prophet(df, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
                national_holidays_abbreviation=prophet_country, future_dataframe_periods=14, freq=prophet_freq, seasonality_mode='additive')
                 #transform media variables into dim return and adstock
        df_saturation_ridge,ridge_intercept,ridge_coefs=transformations_parameters(df, all_media, parameters, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept)
        #add the coefficients into the function
        result, model=ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value, size=0.2, positive=True, random_state=42, coeffs=ridge_coefs[ : -1], intercept=ridge_intercept)
        nrmse=show_nrmse(result[dep_var], result['prediction'])
        decomp_rssd=rssd(df, ridge_coefs, verbose=False)
        print('NRMSE', nrmse)
        print('decomp_rssd', decomp_rssd)
        return nrmse, decomp_rssd


    parameters=create_parametrization(summary_dict)
    instrum = ng.p.Instrumentation(**parameters)
    optimizer= ng.optimizers.TwoPointsDE(instrum, budget=50)
    recommendation = optimizer.minimize(objective_func)
    recom = [col.value[1] for col in optimizer.pareto_front()]
    print(recom[0])


    result, model, summary_dict,df_saturation_ridge,lambda_value,nrmse, decomp_rssd=model_refreshed(**recom[0])
    return result, model, summary_dict,df_saturation_ridge,lambda_value,nrmse, decomp_rssd,recom

result, model, summary_dict,df_saturation_ridge,lambda_value,nrmse, decomp_rssd,recom=refresh_model(df,summary_dict, lambda_value,df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
                national_holidays_abbreviation=prophet_country, future_dataframe_periods=14, freq=prophet_freq, seasonality_mode='additive')



print('____' )
print('nrmse', nrmse)
print('____')
print('decomp_rssd', decomp_rssd)
print('____')
print('result', recom)
