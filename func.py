import numpy as np
import warnings
from scipy.stats import weibull_min
from prophet import Prophet
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import nlopt



def import_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive', ridge_size=0.2, ridge_positive=True, ridge_random_state=42):
    warnings.filterwarnings("ignore")
    
    date_var = json_model['InputCollect']['date_var'][0]
    dep_var = json_model['InputCollect']['dep_var'][0]
    dep_var_type = json_model['InputCollect']['dep_var_type'][0]
    if 'prophet_vars' in json_model['InputCollect']:
        prophet_vars = json_model['InputCollect']['prophet_vars']
    else:
        prophet_vars = []
    if 'prophet_country' in json_model['InputCollect']:
        prophet_country = json_model['InputCollect']['prophet_country'][0]
    else:
        prophet_country = 'IT'
    day_interval = json_model['InputCollect']['dayInterval'][0]
    interval_type = json_model['InputCollect']['intervalType'][0]
    window_start = json_model['InputCollect']['window_start'][0]
    window_end = json_model['InputCollect']['window_end'][0]
    paid_media = json_model['InputCollect']['paid_media_spends']
    if 'organic_vars' in json_model['InputCollect']:
        organic = json_model['InputCollect']['organic_vars'][0]
    else:
        organic = []
    all_media = json_model['InputCollect']['all_media']
    all_vars = json_model['InputCollect']['all_ind_vars']
    
    if interval_type == 'day':
        prophet_freq='D'
    else:
        prophet_freq='W'

    df_window = df.loc[(df[date_var] >= window_start) & (df[date_var] <= window_end)]
    df_window.reset_index(inplace=True, drop=True)

    ####################################################SECTION PROPHET####################################################
    
    df_copy = df.copy()

    df_prophet = prophet(df_copy, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
            national_holidays_abbreviation=prophet_country, future_dataframe_periods=prophet_future_dataframe_periods, freq=prophet_freq, seasonality_mode=prophet_seasonality_mode)

    ####################################################SECTION COEFF AND HYPERPARAMETERS####################################################
    
    #initialize lamba for ridge regression
    lambda_value=0

    #take a coef
    coef_dict = {}

    for elem in json_model['ExportedModel']['summary']:
        for key, value in elem.items():
            coef_dict[elem['variable']] = elem['coef']

    #take a hyperparameters
    hyper_dict = {}

    for key, value in json_model['ExportedModel']['hyper_values'].items():
        if key != 'lambda':
            hyper_dict[key] = value[0]
        else:
            lambda_value = value[0]

    #create a summary dic with the coefs and hyperparameters
    default_summary = {'coef': 0, 'alphas': 0, 'gammas': 0, 'shapes': 0, 'scales': 0 }
    summary_dict = {}

    for var in all_vars:
        if var + '_alphas' in list(hyper_dict.keys()):
            summary_dict[var] = dict({'coef': coef_dict[var], 'alphas': hyper_dict[var + '_alphas'], 'gammas': hyper_dict[var + '_gammas'], 'shapes': hyper_dict[var + '_shapes'], 'scales': hyper_dict[var + '_scales'] })
        else:
            summary_dict[var] = dict({'coef': coef_dict[var]})
    
    if '(Intercept)' in list(coef_dict.keys()):
        summary_dict['intercept'] = dict({'coef': coef_dict['(Intercept)']})
        use_intercept = True
    else:
        use_intercept = False
    
    ####################################################SECTION ADSTOCK AND SATURATION FUNCTIONS####################################################
    
    #create PDF weibull function
    #def dweibull(x, shape, scale):    
        #return (shape / scale) * ((x / scale) ** (shape - 1)) * np.exp(-(x / scale) ** shape)
    
    #create Adstock function
    def adstock(x, shape, scale, windlen=None, type="pdf"):
        if windlen is None:
            windlen = len(x)
        if len(x) > 1:
            if type.lower() not in ("cdf", "pdf"):
                raise ValueError("Invalid value for `type`")
            x_bin = np.arange(1, windlen + 1)
            scale_trans = round(np.quantile(x_bin, scale), 0)
            if shape == 0:
                theta_vec_cum = theta_vec = np.zeros(windlen)
            else:
                #if type.lower() == "cdf":
                    #theta_vec = np.concatenate([[1], 1 - stats.weibull_min.cdf(x_bin[:-1], shape=shape, scale=scale_trans)])
                    #theta_vec_cum = np.cumprod(theta_vec)
                if type.lower() == "pdf":
                    theta_vec_cum = _normalize(weibull_min.pdf(x_bin, c=shape, scale=scale_trans))
            x_decayed = [_decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
            x_decayed = np.sum(x_decayed, axis=0)
        else:
            x_decayed = x
            theta_vec_cum = 1
        return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": theta_vec_cum}

    def _normalize(x):
        min_x, max_x = np.min(x), np.max(x)
        if max_x - min_x == 0:
            return np.concatenate([[1], np.zeros(len(x) - 1)])
        else:
            return (x - min_x) / (max_x - min_x)

    def _decay(x_val, x_pos, theta_vec_cum, windlen):
        x_vec = np.concatenate([np.zeros(x_pos - 1), np.full(windlen - x_pos + 1, x_val)])
        theta_vec_cum_lag = list(pd.Series(theta_vec_cum.copy()).shift(periods=x_pos-1, fill_value=0))
        x_prod = x_vec * theta_vec_cum_lag
        return x_prod
    
    #create Saturation function    
    def saturation_hill(x, alpha, gamma, x_marginal=None):
        inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)
        if x_marginal is None:
            x_scurve = x**alpha / (x**alpha + inflexion**alpha)
        else:
            x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
        return x_scurve
    
    ####################################################SECTION CREATE ADSTOCK AND SATURATION DATASET####################################################

    df_adstock = df[all_media].copy()
    for col in df_adstock.columns:
        df_adstock[col] = adstock(df_adstock[col], summary_dict[col]['shapes'], summary_dict[col]['scales'])['x_decayed']
    df_adstock.reset_index(inplace=True, drop=True)
    df_adstock[date_var] = list(df[date_var])


    df_adstock_filtered = df_adstock.loc[(df_adstock[date_var] >= window_start) & (df_adstock[date_var] <= window_end)]
    df_saturation = df_adstock_filtered[all_media].copy()
    for col in df_saturation.columns:
        df_saturation[col] = saturation_hill(df_saturation[col], summary_dict[col]['alphas'], summary_dict[col]['gammas'])
    df_saturation.reset_index(inplace=True, drop=True)
    df_saturation[date_var] = list(df_adstock_filtered[date_var])

    ####################################################SECTION RIlocDGE REGRESSION####################################################
    
    def ridge(df, all_vars, dep_var, lambda_value=0, size=0.2, positive=False, random_state=42, coeffs=[], intercept=0, fit_intercept=True):

        X = df[all_vars]
        y = df[dep_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=random_state)

        model = Ridge(alpha=lambda_value, fit_intercept=fit_intercept, positive=positive)
        model.intercept_ = intercept
        model.coef_ = np.array(coeffs)
        model.fit(X_train, y_train)
        
        model.intercept_ = intercept
        model.coef_ = np.array(coeffs)

        # Ask the model to predict on X_test without having Y_test
        # This will give you exact predicted values

        # We can use our NRMSE and MAPE functions as well

        # Create new DF not to edit the original one
        result = df.copy()

        # Create a new column with predicted values
        result['prediction'] = model.predict(X)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics_values = {}

        return result, model
    
    df_saturation_ridge = df_saturation.copy()
    df_saturation_ridge.reset_index(inplace=True, drop=True)
    
    for var in all_vars: 
        if var not in df_saturation_ridge.columns:
            df_saturation_ridge[var] = list(df_prophet[var])
            
    df_saturation_ridge[dep_var] = list(df_window[dep_var])
    df_saturation_ridge[date_var] = list(df_window[date_var])
    df_saturation_ridge.fillna(0, inplace=True)
    
    ridge_coefs = [value['coef'] for key, value in summary_dict.items()]
    if use_intercept:
        ridge_intercept = ridge_coefs[-1]
    else:
        ridge_intercept = 0
    
    ridge_only_coefs = ridge_coefs[ : -1]

    ridge_result, ridge_model = ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value, size=ridge_size, positive=ridge_positive, random_state=ridge_random_state, coeffs=ridge_only_coefs, intercept=ridge_intercept)

    if use_intercept:
        ridge_result['intercept'] = 1
        all_vars_with_intercept = all_vars.copy()
        all_vars_with_intercept.append('intercept')
        df_alldecomp_matrix = pd.DataFrame(columns=all_vars_with_intercept)
    else:
        df_alldecomp_matrix = pd.DataFrame(columns=all_vars)
    
    for col in df_alldecomp_matrix.columns:
        df_alldecomp_matrix[col] = summary_dict[col]['coef'] * ridge_result[col]
    
        
    df_alldecomp_matrix[date_var] = list(ridge_result[date_var])
    df_alldecomp_matrix[dep_var] = list(ridge_result[dep_var])
    df_alldecomp_matrix['prediction'] = list(ridge_result['prediction'])
    
    return ridge_model, ridge_result, df_alldecomp_matrix, df_adstock, df_saturation, summary_dict

def prophet(df, df_holidays, date_var, dep_var, prophet_vars, window_start='', window_end='',
        national_holidays_abbreviation='IT', future_dataframe_periods=14, freq='D', seasonality_mode='additive'):

    if 'trend' in prophet_vars:
        trend_seasonality=True
    else:
        trend_seasonality=False
    if 'holiday' in prophet_vars:
        holiday_seasonality=True
    else:
        holiday_seasonality=False
    if 'weekday' in prophet_vars:
        weekday_seasonality=True
    else:
        weekday_seasonality=False
    if 'season' in prophet_vars:
        season_seasonality=True
    else:
        season_seasonality=False
    if 'monthly' in prophet_vars:
        monthly_seasonality=True
    else:
        monthly_seasonality=False

    # Select the Holidays according to the country that interests me
    condition = (df_holidays['country'] == national_holidays_abbreviation)
    
    holidays = df_holidays.loc[condition, ['ds', 'holiday']]

    # Create a DF with the only two columns for Prophet
    prophet_df = df[[date_var, dep_var]]

    # Rename the columns for Prophet
    prophet_df = prophet_df.rename(columns={date_var: 'ds', dep_var: 'y'})

    # Instance and fit Prophet
    prophet_m = Prophet(weekly_seasonality=weekday_seasonality, yearly_seasonality=season_seasonality,
                        daily_seasonality=False, holidays=holidays, seasonality_mode=seasonality_mode)
    if monthly_seasonality:
        prophet_m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
    prophet_m.fit(prophet_df)
    
    future = prophet_m.make_future_dataframe(periods=28, freq=freq)

    forecast = prophet_m.predict(future)
    

    new_forecast = forecast[['ds', 'yhat', 'trend', 'holidays', 'additive_terms', 'multiplicative_terms']].copy()

    if 'yearly' in forecast:
        new_forecast['season'] = forecast['yearly'].copy()
    if 'monthly' in forecast:
        new_forecast['monthly'] = forecast['monthly'].copy()
    if 'weekly' in forecast:
        new_forecast['weekday'] = forecast['weekly'].copy()

    sub_prophet_df = new_forecast[['ds']].copy()

    if trend_seasonality:
        sub_prophet_df['trend'] = new_forecast['trend']
    if holiday_seasonality:
        sub_prophet_df['holiday'] = new_forecast['holidays']
    if 'season' in new_forecast:
        sub_prophet_df['season'] = new_forecast['season']
    if 'weekday' in new_forecast:
        sub_prophet_df['weekday'] = new_forecast['weekday']
    if 'monthly' in new_forecast:
        sub_prophet_df['monthly'] = new_forecast['monthly']

    sub_prophet_df = sub_prophet_df.rename(columns={'ds': date_var})

    df[date_var] = pd.to_datetime(df[date_var])
    sub_prophet_df[date_var] = pd.to_datetime(sub_prophet_df[date_var]) 

    full_df = pd.merge(df, sub_prophet_df, how='inner', on=date_var)
    df_window = full_df.loc[(full_df[date_var] >= window_start) & (full_df[date_var] <= window_end)]
    
    return df_window



def values_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive'):
    warnings.filterwarnings("ignore")
    
    date_var = json_model['InputCollect']['date_var'][0]
    dep_var = json_model['InputCollect']['dep_var'][0]
    dep_var_type = json_model['InputCollect']['dep_var_type'][0]
    if 'prophet_vars' in json_model['InputCollect']:
        prophet_vars = json_model['InputCollect']['prophet_vars']
    else:
        prophet_vars = []
    if 'prophet_country' in json_model['InputCollect']:
        prophet_country = json_model['InputCollect']['prophet_country'][0]
    else:
        prophet_country = 'IT'
    day_interval = json_model['InputCollect']['dayInterval'][0]
    interval_type = json_model['InputCollect']['intervalType'][0]
    window_start = json_model['InputCollect']['window_start'][0]
    window_end = json_model['InputCollect']['window_end'][0]
    paid_media = json_model['InputCollect']['paid_media_spends']
    if 'organic_vars' in json_model['InputCollect']:
        organic = json_model['InputCollect']['organic_vars'][0]
    else:
        organic = []
    all_media = json_model['InputCollect']['all_media']
    all_vars = json_model['InputCollect']['all_ind_vars']
    
    if interval_type == 'day':
        prophet_freq='D'
    else:
        prophet_freq='W'

    df_window = df.loc[(df[date_var] >= window_start) & (df[date_var] <= window_end)]
    df_window.reset_index(inplace=True, drop=True)

    ####################################################SECTION PROPHET####################################################
    
    df_copy = df.copy()

    df_prophet = prophet(df_copy, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
            national_holidays_abbreviation=prophet_country, future_dataframe_periods=prophet_future_dataframe_periods, freq=prophet_freq, seasonality_mode=prophet_seasonality_mode)

    ####################################################SECTION COEFF AND HYPERPARAMETERS####################################################
    
    #initialize lamba for ridge regression
    lambda_value=0

    #take a coef
    coef_dict = {}

    for elem in json_model['ExportedModel']['summary']:
        for key, value in elem.items():
            coef_dict[elem['variable']] = elem['coef']

    #take a hyperparameters
    hyper_dict = {}

    for key, value in json_model['ExportedModel']['hyper_values'].items():
        if key != 'lambda':
            hyper_dict[key] = value[0]
        else:
            lambda_value = value[0]

    #create a summary dic with the coefs and hyperparameters
    default_summary = {'coef': 0, 'alphas': 0, 'gammas': 0, 'shapes': 0, 'scales': 0 }
    summary_dict = {}

    for var in all_vars:
        if var + '_alphas' in list(hyper_dict.keys()):
            summary_dict[var] = dict({'coef': coef_dict[var], 'alphas': hyper_dict[var + '_alphas'], 'gammas': hyper_dict[var + '_gammas'], 'shapes': hyper_dict[var + '_shapes'], 'scales': hyper_dict[var + '_scales'] })
        else:
            summary_dict[var] = dict({'coef': coef_dict[var]})
    
    if '(Intercept)' in list(coef_dict.keys()):
        summary_dict['intercept'] = dict({'coef': coef_dict['(Intercept)']})
        use_intercept = True
    else:
        use_intercept = False
    df_adstock = df[all_media].copy()


    return date_var,dep_var,window_start,window_end, all_media,all_vars,df_prophet, summary_dict,lambda_value,df_window,use_intercept,prophet_vars,prophet_country, prophet_freq

#create PDF weibull function
#def dweibull(x, shape, scale):    
    #return (shape / scale) * ((x / scale) ** (shape - 1)) * np.exp(-(x / scale) ** shape)

#create Adstock function
def adstock(x, shape, scale, windlen=None, type="pdf"):
    if windlen is None:
        windlen = len(x)
    if len(x) > 1:
        if type.lower() not in ("cdf", "pdf"):
            raise ValueError("Invalid value for `type`")
        x_bin = np.arange(1, windlen + 1)
        scale_trans = round(np.quantile(x_bin, scale), 0)
        if shape == 0:
            theta_vec_cum = theta_vec = np.zeros(windlen)
        else:
            #if type.lower() == "cdf":
                #theta_vec = np.concatenate([[1], 1 - stats.weibull_min.cdf(x_bin[:-1], shape=shape, scale=scale_trans)])
                #theta_vec_cum = np.cumprod(theta_vec)
            if type.lower() == "pdf":
                theta_vec_cum = _normalize(weibull_min.pdf(x_bin, c=shape, scale=scale_trans))
        x_decayed = [_decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
        x_decayed = np.sum(x_decayed, axis=0)
    else:
        x_decayed = x
        theta_vec_cum = 1
    return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": theta_vec_cum}

def _normalize(x):
    min_x, max_x = np.min(x), np.max(x)
    if max_x - min_x == 0:
        return np.concatenate([[1], np.zeros(len(x) - 1)])
    else:
        return (x - min_x) / (max_x - min_x)

def _decay(x_val, x_pos, theta_vec_cum, windlen):
    x_vec = np.concatenate([np.zeros(x_pos - 1), np.full(windlen - x_pos + 1, x_val)])
    theta_vec_cum_lag = list(pd.Series(theta_vec_cum.copy()).shift(periods=x_pos-1, fill_value=0))
    x_prod = x_vec * theta_vec_cum_lag
    return x_prod

#create Saturation function    
def saturation_hill(x, alpha, gamma, x_marginal=None):
    inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + inflexion**alpha)
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
    return x_scurve

def transformations(df, all_media, summary_dict, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept):
    df_adstock = df[all_media].copy()
    for col in df_adstock.columns:
        df_adstock[col] = adstock(df_adstock[col], summary_dict[col]['shapes'], summary_dict[col]['scales'])['x_decayed']
    df_adstock.reset_index(inplace=True, drop=True)
    df_adstock[date_var] = list(df[date_var])


    df_adstock_filtered = df_adstock.loc[(df_adstock[date_var] >= window_start) & (df_adstock[date_var] <= window_end)]
    df_saturation = df_adstock_filtered[all_media].copy()
    for col in df_saturation.columns:
        df_saturation[col] = saturation_hill(df_saturation[col], summary_dict[col]['alphas'], summary_dict[col]['gammas'])
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
    ridge_coefs = [value['coef'] for key, value in summary_dict.items()]
    if use_intercept:
        ridge_intercept = ridge_coefs[-1]
    else:
        ridge_intercept = 0

    return df_saturation_ridge,ridge_intercept,ridge_coefs

def ridge(df, all_vars, dep_var, lambda_value=0, size=0.2, positive=False, random_state=42, coeffs=[], intercept=0, fit_intercept=True):

    X = df[all_vars]
    y = df[dep_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=random_state)

    model = Ridge(alpha=lambda_value, fit_intercept=fit_intercept, positive=positive)
    model.intercept_ = intercept
    model.coef_ = np.array(coeffs)
    model.fit(X_train, y_train)
    
    model.intercept_ = intercept
    model.coef_ = np.array(coeffs)

    # Ask the model to predict on X_test without having Y_test
    # This will give you exact predicted values

    # We can use our NRMSE and MAPE functions as well

    # Create new DF not to edit the original one
    result = df.copy()

    # Create a new column with predicted values
    result['prediction'] = model.predict(X)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_values = {}

    return result, model


def rssd(df, coefs, verbose=False):

    media = [col for col in df.columns if '_spend' in col]

    def get_effect_share(df, coefs):
        def effect_share(contribution_df):
            return (contribution_df.sum() / contribution_df.sum().sum()).values

        contr_df = {}

        for m in media:
            contr_df[m] = df[m] * coefs[media.index(m)]

        contr_df = pd.DataFrame.from_dict(contr_df)
        ef_share = effect_share(contr_df)

        return ef_share

    def get_spend_share(df):
        def spend_share(X_df):
            return (X_df.sum() / X_df.sum().sum()).values

        spend_df = {}

        for m in media:
            spend_df[m] = df[m]

        spend_df = pd.DataFrame.from_dict(spend_df)
        ss_share = spend_share(spend_df)

        return ss_share

    value = round(np.sqrt(sum((np.array(get_effect_share(df, coefs)) - np.array(get_spend_share(df))) ** 2)), 3)
    passed = "✔️" if value < 0.15 else "❌"

    
    return value

def show_nrmse(y_actual, y_pred, verbose=False):
    value = np.sqrt(np.mean((y_actual - y_pred) ** 2)) / (max(y_actual) - min(y_actual))
    passed = "✔️" if value < 0.15 else "❌"
    if verbose:
        return value, passed
    else:
        return value

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
    """
def robyn_model(df, all_media, summary_dict, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept,prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive'):

    df_prophet = prophet(df, df_holidays, date_var, dep_var, prophet_vars, window_start=window_start, window_end=window_end,
            national_holidays_abbreviation=prophet_country, future_dataframe_periods=prophet_future_dataframe_periods, freq=prophet_freq, seasonality_mode=prophet_seasonality_mode)

    df_saturation_ridge,ridge_intercept,ridge_coefs= transformations(df, all_media, summary_dict, window_start, window_end,date_var,df_prophet,all_vars,dep_var,df_window,use_intercept)


    result, model=ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value, size=0.2, positive=True, random_state=42, coeffs=ridge_coefs[ : -1], intercept=ridge_intercept)

    Nrmse=nrmse(result[dep_var], result['prediction'])

    return result, model, summary_dict,df_saturation_ridge,lambda_value,Nrmse
"""

def budget_allocation(spends, lb, ub, maxeval, xtol_rel, eval_list, algoritm='GN_ISRES'):

    expSpendUnitTotal = eval_list["expSpendUnitTotal"]

    def eval_f(_spends, grad=[]):
        X = _spends.copy()
        coefsFiltered = eval_list["coefsFiltered"]
        alphas = eval_list["alphas"]
        gammaTrans = eval_list["gammaTrans"]

        def fx_objective(x, coeff, alpha, gammaTran):
            xAdstocked = x
            xOut = coeff * ((1 + gammaTran**alpha / xAdstocked**alpha)**-1)
            return xOut 

        def objective(_X, _coefsFiltered, _alphas, _gammaTrans):
            return -sum(([fx_objective(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(spends, coefsFiltered, alphas, gammaTrans)]))


        def fx_gradient(x, coeff, alpha, gammaTran):
            xAdstocked = x
            xOut = -coeff * ((alpha * (gammaTran**alpha) * (xAdstocked**(alpha - 1))) / (xAdstocked**alpha + gammaTran**alpha)**2)
            return xOut

        def gradient(X, coefsFiltered, alphas, gammaTrans):
            return [fx_gradient(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)]

        '''optm = {
            'objective' : -sum(([fx_objective(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)])),
            'gradient' : [fx_gradient(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)], 
            'objective_channel' : [fx_objective_chanel(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)]
        }'''
        grad = [fx_gradient(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)] 
        return -sum(([fx_objective(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)]))

    if algoritm == 'LN_COBYLA':
        opt = nlopt.opt(nlopt.LN_COBYLA, len(spends))

    elif algoritm == 'GN_ISRES':
        opt = nlopt.opt(nlopt.GN_ISRES, len(spends))

    elif algoritm == 'GN_AGS':
        opt = nlopt.opt(nlopt.GN_AGS, len(spends))

    elif algoritm == 'LD_SLSQP':
        opt = nlopt.opt(nlopt.LD_SLSQP, len(spends))

    elif algoritm == 'LD_AUGLAG':
        opt = nlopt.opt(nlopt.LD_AUGLAG, len(spends))
    
    elif algoritm == 'LD_MMA':
        opt = nlopt.opt(nlopt.LD_MMA, len(spends))

    else:
        opt = nlopt.opt(nlopt.LN_COBYLA, len(spends))

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)

    opt.set_min_objective(eval_f)
    opt.add_inequality_constraint(lambda z, grad: sum(z) - expSpendUnitTotal, xtol_rel)
    opt.add_equality_constraint(lambda z, grad: sum(z) - expSpendUnitTotal, xtol_rel)

    # rate of improvement, below which we are done
    budget_spends = opt.optimize(spends)

    return budget_spends




