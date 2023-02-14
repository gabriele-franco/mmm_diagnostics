from func import import_model, values_model,transformations,ridge, prophet
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error
import nevergrad as ng
from func import adstock, saturation_hill


def transformations_parameters(df, all_media, summary_dict, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept):
    df_adstock = df[all_media].copy()
    for col in df_adstock.columns:
        df_adstock[col] = adstock(df_adstock[col], summary_dict[f'{col}_shapes'], summary_dict[f'{col}_scales'])['x_decayed']
        
 
    df_adstock.reset_index(inplace=True, drop=True)
    df_adstock[date_var] = list(df[date_var])


    df_adstock_filtered = df_adstock.loc[(df_adstock[date_var] >= window_start) & (df_adstock[date_var] <= window_end)]
    df_saturation = df_adstock_filtered[all_media].copy()
    for col in df_saturation.columns:
        df_saturation[col] = saturation_hill(df_saturation[col], summary_dict[f'{col}_alphas'], summary_dict[f'{col}_gammas'])
    df_saturation.reset_index(inplace=True, drop=True)
    df_saturation[date_var] = list(df_adstock_filtered[date_var])
    df_saturation_ridge = df_saturation.copy()
    df_saturation_ridge.reset_index(inplace=True, drop=True)
    
    for var in all_vars: 
        if var not in df_saturation_ridge.columns:
            df_saturation_ridge[var] = list(df_prophet[var])
            
    df_saturation_ridge[dep_var] = list(df_window[dep_var])
    df_saturation_ridge[date_var] = list(df_window[date_var])
    df_saturation_ridge.fillna(0, inplace=True)
    ridge_coefs=[summary_dict.get(col) for col in summary_dict.keys() if 'coef' in col]
    if use_intercept:
        ridge_intercept = ridge_coefs[-1]
    else:
        ridge_intercept = 0

    return df_saturation_ridge,ridge_intercept,ridge_coefs

    
def create_parametrization(summary_dict):
    parameters = {}
    for key, value in summary_dict.items():
        parameters[key] = {}
        for i in value:
            if summary_dict[key][i] != 0:
                parameters[key][i]=ng.p.Scalar(lower=summary_dict[key][i]*0.8, upper=summary_dict[key][i]*1.2)
            else:
                parameters[key][i]=ng.p.Scalar(lower=0, upper=20)
    return parameters

"""

parameters=create_parametrization(summary_dict)

instrum = ng.p.Instrumentation(**parameters)
print(instrum)
optimizer= ng.optimizers.TwoPointsDE(instrum, budget=100)
recommendation = optimizer.minimize(robyn_model_obj)
print(**recommendation)
"""



def create_parametrization(summary_dict):
    parameters = {}
    
    for key, value in summary_dict.items():
        if 'alphas' in value:
            parameters[f'{key}_alphas']=ng.p.Scalar(lower=summary_dict[key]['alphas']*0.8, upper=summary_dict[key]['alphas']*1.2)
        if 'gammas' in value:
            parameters[f'{key}_gammas']=ng.p.Scalar(lower=summary_dict[key]['gammas']*0.8, upper=summary_dict[key]['gammas']*1.2)
        if 'shapes' in value:
            parameters[f'{key}_shapes']=ng.p.Scalar(lower=summary_dict[key]['shapes']*0.8, upper=summary_dict[key]['shapes']*1.2)
        if 'scales' in value:
            parameters[f'{key}_scales']=ng.p.Scalar(lower=summary_dict[key]['scales']*0.8, upper=summary_dict[key]['scales']*1.2)
        if 'coef' in value:
            if summary_dict[key]['coef'] != 0:
                parameters[f'{key}_coef']=ng.p.Scalar(lower=summary_dict[key]['coef']*0.5, upper=summary_dict[key]['coef']*1.5)
            else:
                parameters[f'{key}_coef']=ng.p.Scalar(lower=0, upper=20)
    
    return parameters
"""def create_parametrization(summary_dict):
    parameters = []
    
    for key, value in summary_dict.items():
        if 'alphas' in value:
            parameters.append(ng.p.Scalar(lower=summary_dict[key]['alphas']*0.8, upper=summary_dict[key]['alphas']*1.2).set_mutation(sigma=0.5).set_name(f"{key}_alphas"))
        if 'gammas' in value:
            parameters.append(ng.p.Scalar(lower=summary_dict[key]['gammas']*0.8, upper=summary_dict[key]['gammas']*1.2).set_mutation(sigma=0.5).set_name(f"{key}_gammas"))
        if 'shapes' in value:
            parameters.append(ng.p.Scalar(lower=summary_dict[key]['shapes']*0.8, upper=summary_dict[key]['shapes']*1.2).set_mutation(sigma=0.5).set_name(f"{key}_shapes"))
        if 'scales' in value:
            parameters.append(ng.p.Scalar(lower=summary_dict[key]['scales']*0.8, upper=summary_dict[key]['scales']*1.2).set_mutation(sigma=0.5).set_name(f"{key}_scales"))
        if 'coef' in value:
            if summary_dict[key]['coef'] != 0:
                parameters.append(ng.p.Scalar(lower=summary_dict[key]['coef']*0.8, upper=summary_dict[key]['coef']*1.2).set_mutation(sigma=0.5).set_name(f"{key}_coef"))
            else:
                parameters.append(ng.p.Scalar(lower=0, upper=20))
    
    return parameters
"""