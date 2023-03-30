import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def carry_out_iterations( data,list_of_cols,t_w,id_colname): 
    
    max_t_w_value =  0.975
    train_data =  data[list_of_cols].copy()

    ## creating empty data frame with same structure as traindata to copy confidence scores 
    train_data_confidence =  train_data.copy()
    train_data_confidence.loc[:,:]= 0

    ## calculating (1-t(w)). Carrying out calculation required for the equation
    t_w_inv =  1- t_w
    tau_w =  -np.log(t_w_inv)

    ## creating dataframe that maintains list of confidence values through each iteration
    confidence_iterations = pd.DataFrame(columns =train_data.columns.tolist() + ['iteration'])
    t_w_df = pd.DataFrame(columns = train_data.columns)

    for iteration in range(0,70):

        for col_name in list_of_cols:
            column_matching_df=  train_data_confidence.copy()
            column_matching_df.loc[:,:]= 0
            current_source =  train_data[col_name]

            other_sources_cols = [x for x in list_of_cols if x != current_source.name]

            column_matching_df[col_name] = 1
            for col_name_others in other_sources_cols:
                column_matching_df[col_name_others] = np.where(train_data[col_name_others]==current_source,1,-1)
            column_matching_df[pd.isnull(train_data)]=0

            for col_ii in range(0,column_matching_df.shape[1]):
                column_matching_df.iloc[:,col_ii] = column_matching_df.iloc[:,col_ii] * tau_w[col_ii]

            train_data_confidence[col_name]= np.where(pd.isnull(current_source),np.nan,1/(1 + np.exp(-column_matching_df.sum(axis=1))))


        ## maintaining record of the trusworthiness scores of websites
        t_w_prev =  t_w.copy()
        t_w_df.loc[iteration]= t_w
        t_w = train_data_confidence.mean()
        t_w [t_w >= max_t_w_value] = max_t_w_value
        t_w_inv =  1- t_w
        tau_w =  -np.log(t_w_inv)

        ## printing itertion number and the trustworthiness score
        print(iteration, np.array(t_w_prev))
        if iteration > 5:
            if np.sum(np.abs(t_w.values - t_w_prev.values)) < 0.001:
                break
    
    train_data_confidence[id_colname] =  data[id_colname]
    
    return(t_w_df,train_data_confidence )

def get_final_confidence(data, column_to_check_confidence ,train_data_confidence ,id_colname):
    
    data['final_confidence'] = 0
    all_source_columns = [x for x in train_data_confidence.columns if x != id_colname]
    
    for col_name_source in all_source_columns:
        matching_rows = data[column_to_check_confidence]==data[col_name_source]
        data.loc[matching_rows,'final_confidence'] = train_data_confidence.loc[matching_rows,col_name_source]
        
    return(data)

def perform_operation(data):
    column_to_check_confidence = 'Krushak_Odisha'
    list_of_cols =['Krushak_Odisha','Source_A','Source_B','Source_C','Source_D']
    no_cols =  len(list_of_cols)
    t_w = np.repeat(0.5,no_cols)
    id_colname = 'id'
    t_w_df,train_data_confidence = carry_out_iterations( data,list_of_cols,t_w,id_colname)
    df = get_final_confidence(data, column_to_check_confidence ,train_data_confidence ,id_colname)
    return df

def check_editable(c):
    if c in ["id","final_confidence"]:
        return False
    return True

def find_change_column(df,df_2):
    df_2 = df_2.fillna("")
    df = df.fillna("")
    for col in list(df.columns):
        if (df[col]!=df_2[col]).sum() != 0:
            return col
    return None