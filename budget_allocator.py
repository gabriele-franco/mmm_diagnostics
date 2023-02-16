
from func_budget_allocator import values_model, budget_allocation,eval_f
import numpy as np
import pandas as pd
import streamlit as st
import os

json_model=pd.read_json('RobynModel-7_946_5.json')

csv_files = [f for f in os.listdir("./") if f.endswith(".csv")]

selected_file = st.sidebar.selectbox("Select a CSV file", csv_files)

if selected_file is not None:
    df=pd.read_csv(selected_file)
    st.dataframe(df)

df_holidays=pd.read_csv('dataset_holidays.txt', delimiter=',')

date_var,dep_var,window_start,window_end, all_media,all_vars, summary_dict,lambda_value,df_window,use_intercept,prophet_vars,prophet_country, prophet_freq,paid_media,df_adstock=values_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive')


eval_list = {
        'coefsFiltered': [summary_dict[col]['coef'] for col in paid_media],
        'alphas':[summary_dict[col]['alphas'] for col in paid_media],
        'gammaTrans': [round(np.quantile(np.arange(min(df_adstock[col]), max(df_adstock[col]), 100), summary_dict[col]['gammas'] ), 4) for col in paid_media],
}

xtol_rel=1e-10


past_weeks=int(st.sidebar.text_input("How many weeks in the past do you want to go?", value=4))


transactions=df['transactions'].iloc[-past_weeks]
spends=df[paid_media].tail(past_weeks)

if st.button(f"click to see how many more conversions you would've generate if you used Cassandra Budget allocator in the past {past_weeks} weeks"):

    investments=[]
    output={}
    distribution={}
    for index, row in spends.iterrows():
        spend= np.array(row)
        tot_spend=spend.sum()
        investments.append(tot_spend)
        lb = spend*0.1
        ub=spend*2
        maxeval=30000
        expected_spend=float(st.sidebar.text_input('expected_spend', value=spend.sum()+10))



        old_transactions=abs(eval_f(spend,eval_list, grad=[]))


        budget_spends=budget_allocation(spend,expected_spend, lb, ub, maxeval, xtol_rel, eval_list, algoritm='GN_ISRES')
        
        new_transactions=abs(eval_f(budget_spends,eval_list, grad=[]))
        distribution[index]={'budget_spends':budget_spends,'new_transactions':new_transactions}
        output[index]={'old_transactions':old_transactions, 'new_transactions':new_transactions}


    df2=pd.DataFrame(output).T
    new_transactions=df2['new_transactions'].sum()
    old_transactions=df2['old_transactions'].sum()
    #concatenated_df = pd.concat([df, df2]).fillna(value=0)
    st.write(sum(investments))
    st.line_chart(df2)
    st.write(f'total new transactions {abs(new_transactions)}')
    st.write(f'total new investments {(sum(investments))}')


    #st.bar_chart(budget_spends)


    optimization_target=(abs(new_transactions)-abs(old_transactions))/abs(old_transactions)
    incremental_sales=abs(new_transactions)-abs(old_transactions)
    new_cpo= (sum(investments)/abs(new_transactions))
    old_cpo=sum(investments)/abs(old_transactions)

    cpo_optimization=(new_cpo-old_cpo)/old_cpo
    st.write(f'old cpo {old_cpo}€')
    st.write(f'new cpo {new_cpo}€')


    st.write(f"if you used cassandra last month you could've generate {incremental_sales} incremental sales with the same budget invested")
    st.write(f"The increase in sales compare to what you obtained is: {optimization_target*100}%")
    st.write(f"Your CPA would've been {new_cpo}€ instead of {old_cpo}€, it would've generae a {cpo_optimization*100}% in CPA")






