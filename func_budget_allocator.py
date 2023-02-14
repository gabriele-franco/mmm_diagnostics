import nlopt
import warnings
import numpy as np
from scipy.stats import weibull_min
import pandas as pd

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


def budget_allocation(spends,expected_spend, lb, ub, maxeval, xtol_rel, eval_list, algoritm='GN_ISRES'):

    expSpendUnitTotal = expected_spend #/ (expected_spend_days / day_interval)

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


    return date_var,dep_var,window_start,window_end, all_media,all_vars, summary_dict,lambda_value,df_window,use_intercept,prophet_vars,prophet_country, prophet_freq,paid_media,df[all_media]




def eval_f(_spends,eval_list, grad=[]):
    X = _spends.copy()

    coefsFiltered = eval_list["coefsFiltered"]
    alphas = eval_list["alphas"]
    gammaTrans = eval_list["gammaTrans"]

    def fx_objective(x, coeff, alpha, gammaTran):
        xAdstocked = x
        xOut = coeff * ((1 + gammaTran**alpha / xAdstocked**alpha)**-1)

        return xOut 

    def objective(_X, _coefsFiltered, _alphas, _gammaTrans):
        return -sum(([fx_objective(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(_spends, coefsFiltered, alphas, gammaTrans)]))


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
    prediction=sum(([fx_objective(x, coeff, alpha, gammaTran) for x, coeff, alpha, gammaTran in zip(X, coefsFiltered, alphas, gammaTrans)]))
    return -prediction
