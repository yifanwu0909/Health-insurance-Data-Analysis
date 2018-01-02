# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:56:29 2017

@author: lifen
"""

import requests
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import networkx as nx
import community
from apyori import apriori
import matplotlib.pyplot as plt
import plotly
plotly.tools.set_credentials_file(username='lifengdi', api_key='4atIfbTtxieiOj2T0UQn')
import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import scipy.stats as stats
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D



def Data_Collection():
    def df1_Collect():
        # Endpoint of the first dataset
        BaseURL = 'https://api.census.gov/data/timeseries/healthins/sahie'
        
        # Parameters that we selected to collect (Part A, year = 2015)
        URLPost = {'get': 'NAME,NIC_PT,NIC_LB90,NIC_UB90,NUI_PT,NUI_LB90,NUI_UB90',
                   'for': 'county:*',
                   'in': 'state:*',
                   'time': '2015'
                   }
        
        # Scrape data with a json format
        response = requests.get(BaseURL, URLPost)
        jsontxt = response.json()
        
        # Initialize a dataframe with column names assigned
        df = pd.DataFrame(columns = ('County_State','Number_Insured','NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured',
                                     'NUninsured_CI_LowerBound','NUninsured_CI_UpperBound','time'))
        
        # Append data to each column of the dataframe, skip the column name row
        for i in range(1,len(jsontxt)):
            Location = jsontxt[i][0]
            NInsured = jsontxt[i][1]
            NIC_LB90 = jsontxt[i][2]
            NIC_UB90 = jsontxt[i][3]
            NUnisured = jsontxt[i][4]
            NUI_LB90 = jsontxt[i][5]
            NUI_UB90 = jsontxt[i][6]
            time = jsontxt[i][7]
            df = df.append({'County_State':Location,'Number_Insured':NInsured,'NInsured_CI_LowerBound':NIC_LB90,'NInsured_CI_UpperBound':NIC_UB90,
                            'Number_Uninsured':NUnisured,'NUninsured_CI_LowerBound':NUI_LB90,'NUninsured_CI_UpperBound':NUI_UB90,'time':time},ignore_index=True)
        
        # Parameters that we selected to collect (Part B, year = 2013)
        URLPost = {'get': 'NAME,NIC_PT,NIC_LB90,NIC_UB90,NUI_PT,NUI_LB90,NUI_UB90',
                   'for': 'county:*',
                   'in': 'state:*',
                   'time': '2013'
                   }
        
        # Scrape data with a json format
        response = requests.get(BaseURL, URLPost)
        jsontxt = response.json()
        
        # Continue appending data to each column of the dataframe, skip the column name row
        for i in range(1,len(jsontxt)):
            Location = jsontxt[i][0]
            NInsured = jsontxt[i][1]
            NIC_LB90 = jsontxt[i][2]
            NIC_UB90 = jsontxt[i][3]
            NUnisured = jsontxt[i][4]
            NUI_LB90 = jsontxt[i][5]
            NUI_UB90 = jsontxt[i][6]
            time = jsontxt[i][7]
            df = df.append({'County_State':Location,'Number_Insured':NInsured,'NInsured_CI_LowerBound':NIC_LB90,'NInsured_CI_UpperBound':NIC_UB90,
                            'Number_Uninsured':NUnisured,'NUninsured_CI_LowerBound':NUI_LB90,'NUninsured_CI_UpperBound':NUI_UB90,'time':time},ignore_index=True)    
        return df
    
    def df2_Collect():
        # Endpoint of the second dataset    
        BaseURL1 = 'https://data.medicare.gov/resource/2kat-xip9.json'
        
        # Parameters that we selected
    #    URLPost1 = {'$limit':'20000'}
        URLPost1 = {'$limit':'20000'}
        
        # Scrape data with a json format
        response1 = requests.get(BaseURL1, URLPost1)
        jsontxt1 = response1.json()
        
        # Initialize a dataframe with column names assigned
        df1 = pd.DataFrame(columns=('Provider_id','County','State','Lower_Payment_Est','Ave_Payment','Higher_Payment_Est', 'Measure_id'))
        
        # Append data to each column of the dataframe
        for i in jsontxt1:
            try:
                County = i['county_name'] # Deal with data without 'county_name' dictionary key
            except:
                County = np.NaN # Deal with data without 'county_name' dictionary key
            State = i['location_state']
            Lower_Est = i['lower_estimate']
            Higher_Est = i['higher_estimate']
            Payment = i['payment']
            ID = i['provider_id']
            Measure = i['measure_id']
            df1 = df1.append({'Provider_id':ID,'County':County,'State':State,'Lower_Payment_Est':Lower_Est,'Ave_Payment':Payment,
                              'Higher_Payment_Est':Higher_Est,'Measure_id':Measure},ignore_index=True)
    
        return df1
    
    # load the datasets
    DF1 = df1_Collect()
    DF2 = df2_Collect()
    
    return DF1, DF2

DF1, DF2 = Data_Collection()

#%% Preprocessing
def Data_Preprocessing(DF1, DF2): 
    # =================================== Data Cleanliness ============================
    # This section cheack the cleaniness of each dataset
    
    def No_Mid_Check(a, b, c):
        index = list(map(lambda x, y, z: (y < x) and (y > z), a, b, c))
        return sum(index)
    
    State_List = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
                  'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
                  'WI','WY','DISTRICT OF COLUMBIA', 'DC']
    
    def Cleanliness(df):
        
        # Replace all format of missing value by np.NAN    
        df = df.replace(('N/A','Not Available','',' '), np.NAN)
        # Fraction of missing values of each attributes
        Frac_NAN = df.isnull().sum(axis=0)/len(df)
    
        # Fraction of noise value of each attributes (exclude the missing values)
    #    df_N_1 = sum(df['County_State'].dropna().apply(lambda x: len(x.split(', ', 1))) != 2)/len(df) # detect any value cannot split into county and state
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x == 'District of Columbia'))/len(df) # detect 'District of Columbia', which cannot be splited into county and state
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x.split(', ', 1)[-1].upper() not in State_List))/len(df) + df_N_1 #detect any state not exsits
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x.lower().find(' parish') >= 0))/len(df) + df_N_1
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x.lower().find(' county') >= 0))/len(df) + df_N_1
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x.lower().find(' census area') >= 0))/len(df) + df_N_1
        df_N_1 = sum(df['County_State'].dropna().apply(lambda x: x.lower().find('st.') >= 0))/len(df) + df_N_1
        df_N_2 = sum(df['Number_Insured'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_2 = No_Mid_Check(df.NInsured_CI_LowerBound, df.Number_Insured, df.NInsured_CI_UpperBound)/len(df)+df_N_2 # decect the average number is out of the upper and lower bound
        df_N_3 = sum(df['NInsured_CI_LowerBound'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_4 = sum(df['NInsured_CI_UpperBound'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_5 = sum(df['Number_Uninsured'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_5 = No_Mid_Check(df.NUninsured_CI_LowerBound, df.Number_Uninsured, df.NUninsured_CI_UpperBound)/len(df)+df_N_5
        df_N_6 = sum(df['NUninsured_CI_LowerBound'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_7 = sum(df['NUninsured_CI_UpperBound'].dropna().apply(lambda x: not str(x).isdigit()))/len(df) # detect any value that doesn't only contains digits
        df_N_8 = sum(df['time'].dropna().apply(lambda x: x not in ['2015','2013']))/len(df) # detect any value other than '2015' or '2013', which we set in the URLPost
        
        # Store the fractions into a serie
        Frac_Noise = pd.Series((df_N_1,df_N_2,df_N_3,df_N_4,df_N_5,df_N_6,df_N_7,df_N_8), index=('County_State','Number_Insured',
                               'NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured','NUninsured_CI_LowerBound',
                               'NUninsured_CI_UpperBound','time'))
        
        # General cleanliness for each attributes by category in a dataframe format and print it
        Result = pd.DataFrame({'MissingValue':Frac_NAN,'NoiseValue':Frac_Noise})    
        print('\n',Result)
        
        # Equal weight for all the attributes and cleanliness categories, and scale it to a 0-100 range
        Score = 100*(1-(Result.MissingValue.sum()+Result.NoiseValue.sum())/(2*len(Result)))
        print('\nCleanliness Score is',Score)
    
    
    def Cleanliness1(df1):
        # Replace all format of missing value by np.NAN
        df1 = df1.replace(('N/A','Not Available','',' '), np.NAN)
        # Fraction of missing values of each attribute
        Frac_NAN1 = df1.isnull().sum(axis=0)/len(df1)
        
        # Fraction of noise value of each attributes (exclude the missing values)
        df1_N_1 = sum(df1['Provider_id'].dropna().apply(lambda x: (not x.isdigit()) or (len(x) != 6)))/len(df1) # detect any value without a 6-digit format
        df1_N_2 = sum(df1['County'].dropna().apply(lambda x: not any((y.isalpha() or y.isspace()) for y in x)))/len(df1) # detect any value that doesn't only contains digit and alpha
        df1_N_3 = sum(df1['State'].apply(lambda x: (not x.isalpha()) or len(x) != 2))/len(df1) # detect any value without a 2-alpha format
        df1_N_4 = sum(df1['Lower_Payment_Est'].dropna().apply(lambda x: any(y.isalpha() for y in str(x))))/len(df1) # detect any value contains alpha
        df1_N_5 = sum(df1['Ave_Payment'].dropna().apply(lambda x: any(y.isalpha() for y in str(x))))/len(df1) # detect any value contains alpha
        df1_N_5 = No_Mid_Check(df1.Lower_Payment_Est, df1.Ave_Payment, df1.Higher_Payment_Est)/len(df1)+df1_N_5
        df1_N_6 = sum(df1['Higher_Payment_Est'].dropna().apply(lambda x: any(y.isalpha() for y in str(x))))/len(df1) # detect any value contains alpha
        df1_N_9 = sum(df1['Measure_id'].dropna().apply(lambda x: x.upper() not in ['PAYM_30_PN','PAYM_30_HF','PAYM_90_HIP_KNEE','PAYM_30_AMI']))/len(df1) # detect any value not in the list    
            
        # Store the fractions into a serie
        Frac_Noise1 = pd.Series((df1_N_1,df1_N_2,df1_N_3,df1_N_4,df1_N_5,df1_N_6,df1_N_9), index=('Provider_id','County','State',
                               'Lower_Payment_Est','Ave_Payment','Higher_Payment_Est','Measure_id'))
    
        # General cleanliness for each attributes by category in a dataframe format and print it
        Result1 = pd.DataFrame({'MissingValue':Frac_NAN1,'NoiseValue':Frac_Noise1})
        print('\n',Result1)
        
        # Equal weight for all the attributes and cleanliness categories, and scale it to a 0-100 range
        Score1 = 100*(1-(Result1.MissingValue.sum()+Result1.NoiseValue.sum())/(2*len(Result1)))
        print('\nCleanliness Score is',Score1)
        
    #%% =================================== Data Cleaning =============================  
    # This section process the data cleaning
    def Mid_Check(df, a, b, c):
        index = list(map(lambda x, y, z: (y >= x) and (y <= z), a, b, c))
        df = df[index]
        return df
    
    # Replace all format of missing value by np.NAN    
    DF1 = DF1.replace(('N/A','Not Available','',' '), np.NAN)
    DF2 = DF2.replace(('N/A','Not Available','',' '), np.NAN) 
    
    
    def Cleaning_1(DF1):
        # Drop rows with missing values
        df1 = DF1.dropna()   
        # Make all values in lower cases
        df1 = df1.applymap(lambda x: str(x).lower())   
        # Correct 'district of columbia' to 'washington county, dc'
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace('district of columbia', 'washington county, dc'))  
        # Drop rows with wrong county_state values
        df1 = df1[df1['County_State'].apply(lambda x: len(x.split(', ', 1)) == 2)]   
        # Correct ' county'， ' parish'， ' borough', 'st.' in county names to be inconsist with the other dataset
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace(' county', ''))  
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace(' parish', ''))
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace(' borough', ''))
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace('st.', 'saint'))
        df1['County_State'] = df1['County_State'].apply(lambda x: x.replace(' census area', ''))
        # Correct the type of number relative columns to be integer
        df1[['Number_Insured','NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured','NUninsured_CI_LowerBound',
            'NUninsured_CI_UpperBound']] = df1[['Number_Insured','NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured',
                                       'NUninsured_CI_LowerBound','NUninsured_CI_UpperBound']].astype('int64')    
#        # View the boxplot of 
#        df1[['Number_Insured','NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured','NUninsured_CI_LowerBound',
#            'NUninsured_CI_UpperBound']].boxplot()  # The top value belongs to Los Angeles County in 2013 and 2015 year, it is not a wrong value
        # Check the number of insured is within the 90% confidence interval, drop tows with wrong values
        df1 = Mid_Check(df1, df1.NInsured_CI_LowerBound, df1.Number_Insured, df1.NInsured_CI_UpperBound)    
        # Check the number of uninsured is within the 90% confidence interval, drop tows with wrong values
        df1 = Mid_Check(df1, df1.NUninsured_CI_LowerBound, df1.Number_Uninsured, df1.NUninsured_CI_UpperBound)    
        return df1
    
    
    def Cleaning_2(DF2):
        # Drop rows with missing values
        df2 = DF2.dropna()    
        # Make all values in lower cases
        df2 = df2.applymap(lambda x: x.lower())    
        # Delete '$', ',' and ' ' from payment related columns' values 
        translator = lambda x: x.translate(str.maketrans(dict.fromkeys('$, ')))    
        df2[['Lower_Payment_Est','Ave_Payment','Higher_Payment_Est']] = df2[['Lower_Payment_Est','Ave_Payment','Higher_Payment_Est']].applymap(translator)
        # Correct the type of number relative columns to be integer
        df2[['Lower_Payment_Est','Ave_Payment','Higher_Payment_Est']] = df2[['Lower_Payment_Est','Ave_Payment','Higher_Payment_Est']].astype('int64')
        # Check the average payment of a certain care is with the maximum-minimum payment range, drop tows with wrong values
        df2 = Mid_Check(df2, df2.Lower_Payment_Est, df2.Ave_Payment, df2.Higher_Payment_Est)
        return df2
    
    
    df1 = Cleaning_1(DF1)
    df2 = Cleaning_2(DF2)
    
    # Write dataset to a csv file
    with open('Dataset1_Cleaned.csv', 'w') as f:
        df1.to_csv(f,index=False)
    f.close()
    
    with open('Dataset2_Cleaned.csv', 'w') as f:
        df2.to_csv(f,index=False)
    f.close()
    
    with open('Dataset1_Original.csv', 'w') as f:
        DF1.to_csv(f,index=False)
    f.close()
    
    with open('Dataset2_Original.csv', 'w') as f:
        DF2.to_csv(f,index=False)
    f.close()
    
    # =================================== Processing =============================

    def Cleanliness_View(df1, df2, DF1, DF2):

        print('\nBefore cleanning:\n')
        Cleanliness(DF1)
        Cleanliness1(DF2)
        
        print('\nAfter cleanning:\n')
        Cleanliness(df1)
        Cleanliness1(df2)
        
        # Print the results
        print('\nBefore cleanning:\n')
        Cleanliness(DF1)
        Cleanliness1(DF2)
        
        print('\nAfter cleanning:\n')
        Cleanliness(df1)
        Cleanliness1(df2)
    
    Cleanliness_View(df1, df2, DF1, DF2)

    
    #%% =============================================== Further cleaning
    
    # Mutual check the county-state combination, make sure two datasets share a key attribute
    def County_Check(df1, df2):
        # Split the County_State column by the ', ' delimiter
        df1['County'], df1['State'] = zip(*df1['County_State'].apply(lambda x: x.split(', ', 1)))
        # Check the States and Counties in two data sets mutually matched, remove unsupported County or State rows
        df2 = df2[df2['State'].apply(lambda x: x in df1['State'].values)]
        df2 = df2[df2['County'].apply(lambda x: x in df1['County'].values)]
        return df1, df2
    
    df1, df2 = County_Check(df1, df2)    
    
    #%% ===================================== Further cleanning and organizing
    
    def New_Attribute(df1, df2):
        # Add a attribute showing combined county and state name in the second data set, which will be the primary key to merge two data sets together
        df2['County_State'] = df2.County + ', ' + df2.State
        # Add a attribute showing the population
        df1['Population'] = df1.Number_Insured + df1.Number_Uninsured
        # Add a attribute showing the insurance purchase rate
        df1['Insured_rate'] = df1.Number_Insured/df1.Population
        return df1, df2
    
    New_Attribute(df1, df2)
    
    #%% ========================= Restructure the datasets, merge to one dataset
    
    def Combine(df1, df2):
        # Create a copy of the first data set, which only contains counties that the second data set has as well
        df1_1 = df1[df1['County_State'].apply(lambda x: x in df2['County_State'].values)]
        df2_1 = df2[df2['County_State'].apply(lambda x: x in df1['County_State'].values)]
        
        # Restructure datasets, make the combination of county and state a unique value
        # Average the values in 2013 and 2015
        df1_1 = df1_1.groupby(['County_State'], as_index = False).mean()
        # Average the payment values from four types of health treatment
        df2_1 = df2_1.groupby(['County_State', 'Provider_id'], as_index = False).mean()
        # Average the payment values from hosipitals in the same county
        df2_1 = df2_1.groupby(['County_State'], as_index = False).mean()
    
        # Merge two dataset together
        df_all = df1_1.merge(df2_1, how = 'left', on = ['County_State'])
        
        return df_all
    
    df_all = Combine(df1, df2) 
    
    
    #%% ==================================================== Binning
    
    def Binning(df_all):
        df_all['State'] = df_all['County_State'].apply(lambda x: x.split(', ', 1)[-1])
        
        # Create Regions attribute to divide data into 5 parts for 5 regions identified by the National Geographic Society
        # Identify 5 regions
        west = ['wa','or','ca','nv','id','ut','mt','wy','co','ak','hi']
        southwest = ['az','nm','tx','ok']
        midwest = ['nd','sd','ne','ks','mn','ia','mo','wi','il','mi','in','oh']
        southeast = ['ar','la','ms','tn','ky','al','ga','fl','wv','md','dc','va','nc','sc','de']
        northeast = ['pa','nj','ny','ct','ri','ma','vt','me','nh']
        # Divid the combined data set into 5 parts for different regions
        df_all["Regions"] = df_all["State"]
        df_all["Regions"].loc[df_all['State'].isin(west)] = "west"
        df_all["Regions"].loc[df_all['State'].isin(southwest)] = "southwest"
        df_all["Regions"].loc[df_all['State'].isin(midwest)] = "midwest" 
        df_all["Regions"].loc[df_all['State'].isin(southeast)] = "southeast"
        df_all["Regions"].loc[df_all['State'].isin(northeast)] = "northeast"
        df_all["Regions"] = df_all["Regions"].astype('category')
        
        # Check the correlation
    #    Cor = df_all.corr()
        # According the the correlation matrix, we already know some attributes are highly correlated (threshold = 0.85)
        # Create 3 equal-width bins on Insured_rate, Ave_Payment and Population attributes for Association Rule section
        
        def ExtremeValue_Bin(Series):
            Q3 = Series.quantile(q=0.75)
            Q1 = Series.quantile(q=0.25)
            Series.loc[Series > (Q3 + 1.5*(Q3 - Q1))] = (Q3 + 1.5*(Q3 - Q1))
            Series.loc[Series < (Q3 - 1.5*(Q3 - Q1))] = (Q3 - 1.5*(Q3 - Q1))
            return Series
        
        df_all_1 = df_all.copy()
        
        names = ['Very_Low', 'Low', 'Fair', 'High', 'Very_High']
        df_all['Insured_rate_EqWidth'], bins = pd.cut(ExtremeValue_Bin(df_all_1['Insured_rate']), 5, retbins=True, labels = names)
        names = ['Very_Cheap', 'Cheap', 'Moderate', 'Expensive', 'Very_Expensive']
        df_all['Ave_Payment_EqWidth'], bins = pd.cut(ExtremeValue_Bin(df_all_1['Ave_Payment']), 5, retbins=True, labels = names)
        names = ['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
        df_all['Pop_EqWidth'], bins = pd.cut(ExtremeValue_Bin(df_all_1['Population']), 5, retbins=True, labels = names)
        
        return df_all
    
    df_all = Binning(df_all)
    
    def Outlier(df):
        NumAttr = df.select_dtypes(include=['int64', 'float64'])
        Desc = NumAttr.describe().transpose()
        Desc['IQR'] = Desc['75%'] - Desc['25%']
        Desc['Lower_R'] = Desc['25%'] - 1.5*Desc['IQR'] # Upper-boundary of range
        Desc['Upper_R'] = Desc['75%'] + 1.5*Desc['IQR'] # Lower-boundary of range 
    
        for i in NumAttr.columns.values:
            index = list(map(lambda x: (x >= Desc.loc[i,'Lower_R']) and (x <= Desc.loc[i,'Upper_R']), df[i]))
            df = df[index]
    
        return df
    
    df_all = Outlier(df_all)    
    
    # save the dataset into a csv file
    with open('df_all.csv', 'w') as f:
        df_all.to_csv(f, index=False)
    f.close()
    
    return df_all

df_all = Data_Preprocessing(DF1, DF2)

#%% Basic Statistics Analysis
def Basic_Stats(df_all):    
    # Split the dataset into numeric datasets, categorical datasets  
    NumAttr = df_all.select_dtypes(include=['int64', 'float64'])
    CatAttr = df_all.select_dtypes(include=['category'])
    
    # Show min, max, mean, median, and standard deviation of numeric attributes
    Summary1 = pd.DataFrame({'Mean': NumAttr.mean(), 'Median': NumAttr.median(), 'Std': NumAttr.std()})
    # Show mode of categorical attributes
    Summary2 = pd.DataFrame(CatAttr.mode()).transpose() 
    Summary2.columns = ['Mode']

    # Combine the results of two data sets together, and print the results
    print('\nNumeric:\n', Summary1)
    print('\nCategorical:\n', Summary2)
    
#    with open('Num.csv', 'w') as f:
#        Summary1.to_csv(f)
#    f.close()
#    with open('Cat.csv', 'w') as f:
#        Summary2.to_csv(f)
#    f.close()
    
    
Basic_Stats(df_all)


#%% ASSOCIATION RULE=============================================================================================
# Create a function better viewing the frequent sets in a dataframe format
def AssociationRule(df_all):
    def FrequentSet_Viewer(Results):
        df_FSet = pd.DataFrame(columns=('X_Y', 'Support'))
        for i in Results:
            df_FSet = df_FSet.append({'X_Y': str(i[0])[10:-1], 'Support': i[1]}, ignore_index = True)
        print(df_FSet)
#        with open('FSet.csv', 'w') as f:
#            df_FSet.to_csv(f, index = False)
#        f.close()
    
    # Create a function better viewing the association rules in a dataframe format
    def AssociationRule_Viewer(Results):
        df_ARule = pd.DataFrame(columns=('X_Y', 'X', 'Y', 'Support', 'Confidence'))
        for i in Results:
            for j in i[2]:
                df_ARule = df_ARule.append({'X_Y': str(i[0])[10:-1], 'X': str(j[0])[10:-1], 'Y': str(j[1])[10:-1], 'Support': i[1],
                                            'Confidence':j[2]}, ignore_index = True)
        print(df_ARule)
        return df_ARule
    
    
    # To understand characteristics of counties with high insured rate, filter the dataset with high insured rate
    transactions =  df_all.select_dtypes(include=['category']).as_matrix()
    
    
    # Set a minimum support threshold at 0.001, and view the results
    Results_0 = list(apriori(transactions, min_support = 0.2))
    FrequentSet_Viewer(Results_0)
    
    # View the Association Rule results
    Results_0 = list(apriori(transactions, min_support = 0.001))
    ARule0 = AssociationRule_Viewer(Results_0)
    
    #Filter the Assosiation Rule with {X} equals to {'High'}
    ARule1 = ARule0[(ARule0.X != '') & (ARule0.Confidence > 0.75)]
    ARule1 = ARule1[ARule1.Y.str.contains('High')]    
    print(ARule1)
#    with open('ARule1.csv', 'w') as f:
#        ARule1.to_csv(f, index = False)
    
    #Filter the Assosiation Rule with {Y} is {'High'}
    ARule1 = ARule0[(ARule0.X != '') & (ARule0.Confidence > 0.75)]
    ARule1 = ARule1[ARule1.Y.str.contains('Low')]    
    print(ARule1)   
#    with open('ARule2.csv', 'w') as f:
#        ARule1.to_csv(f, index = False)
#    
    
AssociationRule(df_all)

#%% Boxplot

def Boxplot(df_all):
    # Create traces for a boxplot
    trace1 = go.Box(y = df_all[df_all['Regions'] == 'west']['Insured_rate'], name='West', boxmean = True, jitter = .3)
    trace2 = go.Box(y = df_all[df_all['Regions'] == 'southwest']['Insured_rate'], name = "Southwest", boxmean = True, jitter = .3)
    trace3 = go.Box(y = df_all[df_all['Regions'] == 'midwest']['Insured_rate'], name='Midwest', boxmean = True, jitter = .3)
    trace4 = go.Box(y = df_all[df_all['Regions'] == 'southeast']['Insured_rate'], name='Southeast', boxmean = True, jitter = .3)
    trace5 = go.Box(y = df_all[df_all['Regions'] == 'northeast']['Insured_rate'], name='Northeast', boxmean = True, jitter = .3)

    # Assign them to an iterable object named data2
    data = [trace1, trace2, trace3, trace4, trace5]
    
    # Add title
    layout = go.Layout(title = "Box Plots for Insured Rate by Regions")
    
    # Setup figure
    fig = go.Figure(data = data, layout = layout)
    
    #Create the boxplot
    py.plot(fig, filename = 'boxplot')

Boxplot(df_all)


def Histogram(df_all):                        
    # Create traces for a scatterplot
    trace1 = go.Histogram(x = df_all['NInsured_CI_LowerBound'])
    trace2 = go.Histogram(x = df_all['Number_Insured'])
    trace3 = go.Histogram(x = df_all['NInsured_CI_UpperBound'])
    trace4 = go.Histogram(x = df_all['NUninsured_CI_LowerBound'])
    trace5 = go.Histogram(x = df_all['Number_Uninsured'])
    trace6 = go.Histogram(x = df_all['NUninsured_CI_UpperBound'])
    trace7 = go.Histogram(x = df_all['Lower_Payment_Est'])
    trace8 = go.Histogram(x = df_all['Ave_Payment'])
    trace9 = go.Histogram(x = df_all['Higher_Payment_Est'])
#    trace10 = go.Histogram(x = df_all['Population'], xaxis = 'Population')
#    trace11 = go.Histogram(x = df_all['Insured_rate'], xaxis = 'Insured Rate')
    
    # Setup figure
    fig = tools.make_subplots(rows=3, cols=3, subplot_titles = ('Lower CI of Number of Insured', 'Number of Insured', 'Upper CI of Number of Insured', 'Lower CI of Number of Uninsured', 
                                                                'Number of Uninsured', 'Upper CI of Number of Insured', 'Lower_Payment_Est', 'Ave_Payment', 'Higher_Payment_Est'))
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 3)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)
    fig.append_trace(trace6, 2, 3)
    fig.append_trace(trace7, 3, 1)
    fig.append_trace(trace8, 3, 2)
    fig.append_trace(trace9, 3, 3)
#    fig.append_trace(trace10, 4, 1)
#    fig.append_trace(trace11, 4, 2)
    
    fig['layout'].update(title = 'Histogram of Numeric Attributes')
    
    #Create the boxplot
    py.plot(fig, filename = 'histogram')
    
Histogram(df_all)

#%%  Correlation Analysis & Clustering Analysis
def Corr_Cluster(df_all):
    
    def encoding(df_all_wyf, colname):
        for i in colname:
            LabalEncoder = preprocessing.LabelEncoder()
            LabalEncoder.fit(df_all_wyf[i])
            df_all_wyf[i] = pd.Series(LabalEncoder.transform(df_all_wyf[i]))
        return df_all_wyf
        
    #Encode the columns 'Insured_rate_EqWidth','Ave_Payment_EqWidth','Pop_EqWidth','Regions' in df_all_wyf. 
    col=['Insured_rate_EqWidth','Ave_Payment_EqWidth','Pop_EqWidth','Regions']
    df_all_wyf = df_all.copy()
    df_all_wyf = encoding(df_all_wyf, col)

    #A function that plot histogram for selected variables that we want to investigate.
    def getHist(df_all_wyf):
        df_all_wyf[['Number_Insured', 'NInsured_CI_LowerBound','NInsured_CI_UpperBound','Number_Uninsured',
            'NUninsured_CI_LowerBound','NUninsured_CI_UpperBound','Population','Insured_rate',
            'Lower_Payment_Est','Ave_Payment','Higher_Payment_Est','Regions']].hist(layout = (3, 4), figsize = (40, 40))  
    
    #A function that plot histogram for selected variables that we want to investigate using Plotly. 
    def plotlyHist(df_all_wyf):
        #Select the name of the variables that we want to investigate. 
        all_names = ['Number_Insured', 'NInsured_CI_LowerBound','NInsured_CI_UpperBound','Population',
                 'Number_Uninsured','NUninsured_CI_LowerBound','NUninsured_CI_UpperBound','Insured_rate',
            'Lower_Payment_Est','Ave_Payment','Higher_Payment_Est','Regions']
        
        #Draw in Plotly account.
        fig = tools.make_subplots(rows=3, cols=4, subplot_titles=all_names)
        for i in range(len(all_names)):    
            tracei = go.Histogram(x=df_all_wyf[all_names[i]],name = all_names[i])
            fig.append_trace(tracei, int((i+4)/4), i%4+1)
            
        py.iplot(fig, filename='Histograms for all of the attributes from our data')
    
    #Draw histogram for selected variables that we want to investigate.
    getHist(df_all_wyf)
    #Plot histogram for selected variables that we want to investigate using Plotly. 
    plotlyHist(df_all_wyf)
    
    
    #%% Correlation Analysis
    
    #A function that find the correlation between all the paires of the quantity variables in df_all_wyf. 
    def getCorr1(df_all_wyf):
        #Make correlation coefficient data frame. 
        corr_val=df_all_wyf.corr()
        #Round the coefficient in two decimle points. 
        corr_val=corr_val.round(2)
        #Add a name column containing all of our selected variable name for better comparison of the correlation.
        names=corr_val.columns
        corr_val.insert(0, 'names', names, allow_duplicates=False)
        
        corr_val.to_csv('corr_val.csv', sep=',')
        return corr_val
        
    #Plot the correlation table using plotly. 
    def plotlyCorr1(corr_val):
        #Variable selecton:
        names=['names', 'Number_Insured', 'NInsured_CI_LowerBound',
               'NInsured_CI_UpperBound', 'Number_Uninsured',
               'NUninsured_CI_LowerBound', 'NUninsured_CI_UpperBound', 'Population',
               'Insured_rate', 'Lower_Payment_Est', 'Ave_Payment',
               'Higher_Payment_Est','Regions']
        #Create trace for later plotting
        trace = go.Table(
            type = 'table',
            header=dict(values=names,
                        align = 'center'),
            cells=dict(values=[corr_val.names, corr_val.Number_Insured, corr_val.NInsured_CI_LowerBound, corr_val.NInsured_CI_UpperBound, 
                               corr_val.Number_Uninsured, corr_val.NUninsured_CI_LowerBound, 
                               corr_val.NUninsured_CI_UpperBound, corr_val.Population, corr_val.Insured_rate, corr_val.Lower_Payment_Est, corr_val.Ave_Payment, 
                               corr_val.Higher_Payment_Est, corr_val.Regions],
                       align = 'center'))
        
        data = [trace]
        #Draw the trace using plotly. 
        py.iplot(data, filename = 'Correlation table')
       
    #Get histogram for selected number of variables that we trimmed down. 
    def getCorr2(df_all_wyf):
        #Select variables. 
        critical_var=['Ave_Payment','Population', 'Regions','Insured_rate']
        #Make correlation coefficient data frame. 
        corr_val2=df_all_wyf[critical_var].corr()
        #Round the coefficient in two decimle points. 
        corr_val2=corr_val2.round(2)
        #Add a name column for better comparison of the correlation.
        corr_val2.insert(0, 'names', critical_var, allow_duplicates=False)
        return corr_val2
    
    
    def plotlyCorr2(corr_val):
        #Plotly correlation table:
        names1=['names','Ave_Payment','Population', 'Regions','Insured_rate']
        trace1 = go.Table(
            type = 'table',
            header=dict(values=names1,
                        align = 'center'),
            cells=dict(values=[corr_val.names,corr_val.Ave_Payment,corr_val.Population,corr_val.Regions,corr_val.Insured_rate],
                       align = 'center'))
        
        data1 = [trace1] 
        py.iplot(data1, filename = 'Correlation table for critical variables')    
    
    corr_full=getCorr1(df_all_wyf)
    plotlyCorr1(corr_full)
    
    corr_sub=getCorr2(df_all_wyf)
    plotlyCorr2(corr_sub)
    
    #%%Clustering Analysis
       
    #Normalize data frame.
    def normalize(numdata):
        x = numdata.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalizedDataFrame = pd.DataFrame(x_scaled)
        #pprint(normalizedDataFrame[:10])   
        return normalizedDataFrame
    
    # Kmeans function that returns a array of the cluster label in original dataset order. 
    def getKmeans(normalizedDataFrame, k):
        kmeans = KMeans(n_clusters=k)
        cluster_labels_kmeans = kmeans.fit_predict(normalizedDataFrame)
        
        return cluster_labels_kmeans
    
    # Hierarchical Clustering function that returns the linkage matrix.
    def getHierarchical_Z(normalizedDataFrame, fig_num):
        # generate the linkage matrix
        Z = linkage(normalizedDataFrame, 'ward')
        
        def fancy_dendrogram(*args, **kwargs):
            plt.figure(fig_num)
            max_d = kwargs.pop('max_d', None)
            if max_d and 'color_threshold' not in kwargs:
                kwargs['color_threshold'] = max_d
            annotate_above = kwargs.pop('annotate_above', 0)
        
            ddata = dendrogram(*args, **kwargs)
        
            if not kwargs.get('no_plot', False):
                
                # calculate full dendrogram
                plt.title('Hierarchical Clustering Dendrogram (truncated)')
                plt.xlabel('cluster size')
                plt.ylabel('distance')
                for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                    x = 0.5 * sum(i[1:3])
                    y = d[1]
                    if y > annotate_above:
                        plt.plot(x, y, 'o', c=c)
                        plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                     textcoords='offset points',
                                     va='top', ha='center')
                if max_d:
                    plt.axhline(y=max_d, c='k')
            return ddata
        
        fancy_dendrogram(
            Z,
            truncate_mode='lastp',# show only the last p merged clusters
            p=2,# show only the last p merged clusters
            leaf_rotation=90.,# rotates the x axis labels
            leaf_font_size=12.,# font size for the x axis labels
            show_contracted=True,# to get a distribution impression in truncated branches
            annotate_above=10,  # useful in small plots so annotations don't overlap
            )
        plt.show()
    
        return Z
    
    #DBSCAN function tha returns a array of the cluster label in original dataset order.
    def getDBSCAN(normalizedDataFrame,r):  
        dbscan = DBSCAN(eps=r)
        cluster_labels_dbscan = dbscan.fit_predict(normalizedDataFrame)
        return cluster_labels_dbscan
    
    #Print the silhouette score
    def print_silhouette_score( normalizedDataFrame, cluster_labels ):
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)
        
    #Draw the result of the PCA decomposition in a 3D scatterplot. 
    def draw_pca(normalizedDataFrame,cluster_labels,fig_num,plt_name):
        import matplotlib.pyplot as plt
        fig = plt.figure(fig_num) 
        ax = fig.add_subplot(111, projection='3d')#set 3d axes. 
        
        pca3D = decomposition.PCA(3) #PCA decomposition.
        plot_columns = pca3D.fit_transform(normalizedDataFrame)   
        ax.scatter(xs=plot_columns[:,0], ys=plot_columns[:,1],zs=plot_columns[:,2],zdir='z', s=20, c=cluster_labels)
        plt.show()
        
        trace1 = go.Scatter3d(
        x=plot_columns[:,0],
        y=plot_columns[:,1],
        z=plot_columns[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color=cluster_labels,   # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            #opacity=0.8
            )
        )
        data = [trace1]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig, filename=plt_name)
    
    #Concatinate the assigned clusters back to the original dataset and returns two datasets in each cluster. 
    def assign_clusters( numdata, cluster_labels ):
        #Encode the cluster number to 0,1 due to the fact that different clustering machanisms 
        #might label clusters differently. 
        LabalEncoder = preprocessing.LabelEncoder()
        LabalEncoder.fit(cluster_labels)
        cluster_labels = pd.Series(LabalEncoder.transform(cluster_labels))
        #Concatinate the cluster label with the original dataset's copy to get a new dataset. 
        numdata_with_label=numdata
        numdata_with_label['cluster_labels'] = cluster_labels
        #Seperated the labeled dataset into two datasets in each cluster. 
        numdata_label_1=numdata_with_label[numdata_with_label.cluster_labels == 0]
        numdata_label_2=numdata_with_label[numdata_with_label.cluster_labels == 1]
    
        
        return (numdata_label_1, numdata_label_2)
        
    #
    #Draw the distribution of each attributes in two datasets in each cluster. Draw the distribution 
    #of each attributes of two datasets in each cluster together in one histogram. 
    def cluster_distribution( cluster1, cluster2,  fig_num,ply_name):
        colnames=['Number_Insured','Number_Uninsured','Population','Insured_rate',
                             'Ave_Payment','Regions']
        f,a = plt.subplots(2,3)
        plt.figure(fig_num)
        a = a.ravel()

        for i in range(len(colnames)):    
            a[i].hist(cluster1[colnames[i]], normed = 1, alpha=0.8, label='Cluster 1')
            a[i].hist(cluster2[colnames[i]], normed = 1, alpha=0.8, label='Cluster 2')
            a[i].legend(loc='upper right')
            a[i].set_title(colnames[i])
            plt.show()
                    
    #%%       
    #k mean clustering
    #Select the columns that we need to do the clustering on from the encoded dataset. 
    numdata = df_all_wyf.filter(['Number_Insured','Number_Uninsured','Population','Insured_rate','Ave_Payment','Regions'], axis=1)
    #Normalize the dataset as pre-rocessing. 
    normalizedDataFrame=normalize(numdata)    
    #Perform Kmeans clustering on the normalized dataset and get the list of clustering label. 
    cluster_labels_kmeans=getKmeans(normalizedDataFrame,2)
    #Calculate the silhouette score for this clustering method. 
    print_silhouette_score(normalizedDataFrame, cluster_labels_kmeans)
    #Plot the 3D-PCA projection.
    draw_pca(normalizedDataFrame,cluster_labels_kmeans,3,'K-mean PCA')  
    #Seperate the original dataset into two datasets in each cluster.
    numdata_kmeans_label_1, numdata_kmeans_label_2 = assign_clusters( numdata, cluster_labels_kmeans )
    #Draw the distribution of each attributes in two seperated datasets.
    cluster_distribution( numdata_kmeans_label_1, numdata_kmeans_label_2,4,'K-mean histogram')
    
    #%% Hierarchical clustering
    
    # generate the linkage matrix
    Z=getHierarchical_Z(normalizedDataFrame,5)
    #Perform hierarchical clustering on the normalized dataset and get the list of clustering label.
    k=2
    cluster_labels_hierarchical=fcluster(Z, k, criterion='maxclust')
    #Calculate the silhouette score for this clustering method. 
    print_silhouette_score(normalizedDataFrame, cluster_labels_hierarchical)
    #Plot the 3D-PCA projection.
    draw_pca(normalizedDataFrame,cluster_labels_hierarchical,6,'Hierarchical PCA')  
    #Seperate the original dataset into two datasets in each cluster.
    numdata_hierarchical_label_1, numdata_hierarchical_label_2= assign_clusters( numdata, cluster_labels_hierarchical )
    #Draw the distribution of each attributes in two seperated datasets.
    cluster_distribution(numdata_hierarchical_label_1, numdata_hierarchical_label_2,7,'Hierarchical histogram')
    
    
    
    #%%Perform DBSCAN clustering on the normalized dataset and get the list of clustering label.
    cluster_labels_dbscan=getDBSCAN(normalizedDataFrame,0.17)
    #Calculate the silhouette score for this clustering method. 
    print_silhouette_score(normalizedDataFrame, cluster_labels_dbscan)
    #Plot the 3D-PCA projection.
    draw_pca(normalizedDataFrame,cluster_labels_dbscan,8,'DBSCAN PCA')  
    #Seperate the original dataset into two datasets in each cluster.
    numdata_dbscan_label_1, numdata_dbscan_label_2= assign_clusters(numdata, cluster_labels_dbscan)
    #Draw the distribution of each attributes in two seperated datasets.
    cluster_distribution(numdata_dbscan_label_1, numdata_dbscan_label_2,9,'DBSCAN histogram')

Corr_Cluster(df_all)

#%% Machine learning analysis =============================================================================================
def Machine_Learning(df_all):
    def T_test(x,y):
        twosample_results = stats.ttest_ind(x,y, equal_var=False)
        print("Two-way T-Test P = ", twosample_results[1])
    
    # Anova
    def ANOVA(x,y):
         f_val, p_val = stats.f_oneway(x,y)  
         print ("One-way ANOVA P =", p_val  )
    
             
    # Setup 10-fold cross validation to evaluate the accuracy of each model
    # Split data into 10 parts
    # Using cross-validation to each algorithm.
    # Add each algorithm and its name to the model array
    def Evaluate_model (x,y):
        models = []
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('SVM', SVC()))
        models.append(('RF',RandomForestClassifier()))
        models.append(('NB', MultinomialNB()))
        scoring = 'accuracy'
        
        # Evaluate each model, add results to a results array,
        # Print the accuracy results (remember these are averages and std )
        results = []
        names = []
        for name, model in models:
        	kfold = KFold(n_splits = 10, random_state = 7, shuffle=False)
        	cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        	results.append(cv_results)
        	names.append(name)
        	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        	print(msg)
       
    # Decision Tree
    def Decision_Tree(x1, y1,x2,y2,lb):
        y1_orin = lb.inverse_transform(y1)
        CART = DecisionTreeClassifier()
        CART.fit(x1, y1_orin)
        predict_CART = CART.predict(x2)
        y2_orin = lb.inverse_transform(y2)
        print(accuracy_score(y2_orin , predict_CART))   
        print(confusion_matrix(y2_orin, predict_CART))
        print(classification_report(y2_orin , predict_CART))
        return predict_CART
       
    #KNN
    def KNN(x1, y1,x2,y2,lb):
        y1_orin = lb.inverse_transform(y1)
        knn = KNeighborsClassifier()
        knn.fit(x1, y1_orin)
        predict_knn = knn.predict(x2)
        y2_orin = lb.inverse_transform(y2)
        print(accuracy_score(y2_orin, predict_knn))   
        print(confusion_matrix(y2_orin, predict_knn))
        print(classification_report(y2_orin, predict_knn))
        return predict_knn
        
    # Naive Bayes
    def Naive_Bayes(x1, y1,x2,y2,lb):
        y1_orin = lb.inverse_transform(y1)
        gnb = MultinomialNB()
        gnb.fit(x1,y1_orin)
        predict_NB = gnb.predict(x2)
        y2_orin = lb.inverse_transform(y2)
        print(accuracy_score(y2_orin, predict_NB))    
        print(confusion_matrix(y2_orin, predict_NB))
        print(classification_report(y2_orin, predict_NB))
        return predict_NB
        
    # SVM
    def SVM(x1, y1,x2,y2,lb):
        y1_orin = lb.inverse_transform(y1)
        clf = svm.SVC(decision_function_shape='ovr')
        clf.fit(x1, y1_orin)
        predict_SVM = clf.predict(x2) 
        y2_orin = lb.inverse_transform(y2)
        print(accuracy_score(y2_orin, predict_SVM))   
        print(confusion_matrix(y2_orin, predict_SVM))
        print(classification_report(y2_orin, predict_SVM))
        return predict_SVM
       
    # Random Forest
    def Random_Forest(x1, y1,x2,y2,lb):
        y1_orin = lb.inverse_transform(y1)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(x1, y1_orin)
        predict_RF = clf.predict(x2)  
        y2_orin = lb.inverse_transform(y2)
        print(accuracy_score(y2_orin, predict_RF))  
        print(confusion_matrix(y2_orin, predict_RF))
        print(classification_report(y2_orin, predict_RF))
        return predict_RF
        
    #%% Test Hypothesis 1, Insured numbers has nothing to do with regions.
    # Normalize the Number_insured and Number_Uninsured column
    X1 = pd.DataFrame(preprocessing.normalize(df_all[['Number_Insured','Number_Uninsured']]))
    X2 = df_all['Insured_rate']
    # Join the two parts into a objective data
    X = pd.concat([X1,X2], axis=1)
    # Turn the label column into a one-hot format
    Y = df_all.Regions
    lb = preprocessing.LabelBinarizer()
    Y1 = lb.fit_transform(Y)
    
    # Using cross-validation to divide dataset into training data and test data.
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y1, test_size = 0.20, random_state = 7)
    
    # Evaluate the 5 machine learning models to use a suitable one to perform analysis
    Evaluate_model(X_train, lb.inverse_transform(Y_train))
    
    # Perfrom decision tree
    y_score_DT1 = Decision_Tree(X_train, Y_train, X_validate, Y_validate,lb)
    
    ## SVM
    #y_score_SVM1 = SVM(X_train, Y_train, X_validate, Y_validate,lb)    
#    #KNN
#    y_score_knn1 = KNN(X_train, Y_train, X_validate, Y_validate,lb)    
#    # Random Forest
#    y_score_RF1 = Random_Forest(X_train, Y_train, X_validate, Y_validate,lb)    
    ## Naive Bayes
    #y_score_NB1 = Naive_Bayes(X_train, Y_train, X_validate, Y_validate,lb) 
       

    #%% Test Hypothesis 2, Insured numbers has nothing to do with population.
    # Prepare trainning data and validation data
    X2 = df_all["Population"]
    Y2 = df_all["Insured_rate_EqWidth"]
    test_size = 0.20
    seed = 7
    Y2 = lb.fit_transform(Y2)
    X_train2, X_validate2, Y_train2, Y_validate2 = train_test_split(X2, Y2, test_size=test_size, random_state=seed)
    # Correct the data format for later analysis
    X_train2 = pd.DataFrame(X_train2)
    X_validate2 = pd.DataFrame(X_validate2)
    
    # Using cross-validation to each algorithm.
    Evaluate_model(X_train2, lb.inverse_transform(Y_train2))
    
#    # Perfrom decision tree
#    y_score_DT2 = Decision_Tree(X_train2, Y_train2, X_validate2, Y_validate2,lb)
#    ## Perform SVM
#    y_score_SVM2 = SVM(X_train2, Y_train2, X_validate2, Y_validate2,lb)
    
    #KNN
    y_score_knn2 = KNN(X_train2, Y_train2, X_validate2, Y_validate2,lb)
    
    ## Random Forest
    #y_score_RF2 = Random_Forest(X_train2, Y_train2, X_validate2, Y_validate2,lb)    
    ## Naive Bayes
    #y_score_NB2 = Naive_Bayes(X_train2, Y_train2, X_validate2, Y_validate2,lb)
    
    
    #%% Test Hypothesis 3: Insured rate has nothing to do with medical expenditure    
    #T_test
    # The data need to be sampled from the same population
    T_test(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "High"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Low"])
    T_test(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Fair"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Low"])
    T_test(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "High"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Fair"])
       
    # Anova
    ANOVA(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "High"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Low"])
    ANOVA(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "High"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Fair"])
    ANOVA(df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Fair"], df_all["Ave_Payment"][df_all.Insured_rate_EqWidth == "Low"])
       
    # Data Driven predictive models
    # Prepare trainning data and validation data
    X3 = df_all["Ave_Payment"]
    Y3 = df_all["Insured_rate_EqWidth"]
    Y3 = lb.fit_transform(Y3)
    X_train3, X_validate3, Y_train3, Y_validate3 = train_test_split(X3, Y3, test_size = 0.20, random_state = 7)
    # Correct the data format for later analysis
    X_train3 = pd.DataFrame(X_train3)
    X_validate3 = pd.DataFrame(X_validate3)
    
    # Using cross-validation to each algorithm.
    Evaluate_model(X_train3, lb.inverse_transform(Y_train3))
    
#    # Perfrom decision tree
#    y_score_DT3 = Decision_Tree(X_train3, Y_train3, X_validate3, Y_validate3,lb)
    
    ## Perform SVM
    y_score_SVM3 = SVM(X_train3, Y_train3, X_validate3, Y_validate3,lb)
    
    #KNN
    #y_score_knn3 = KNN(X_train3, Y_train3, X_validate3, Y_validate3,lb)    
    ## Random Forest
    #y_score_RF3 = Random_Forest(X_train3, Y_train3, X_validate3, Y_validate3,lb)    
#    ## Naive Bayes
#    y_score_NB3 = Naive_Bayes(X_train3, Y_train3, X_validate3, Y_validate3,lb)
    
    
    #%% ROC Curve
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    def ROC_Curve1(y_test, y_score, n):
        n_classes = n
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("Class" + str(i))
            plt.legend(loc="lower right")
            plt.show()
    
    ROC_Curve1(Y_validate , lb.fit_transform(y_score_DT1), 5)       
    ROC_Curve1(Y_validate2 , lb.fit_transform(y_score_knn2), 5)   
    ROC_Curve1(Y_validate3 , lb.fit_transform(y_score_SVM3), 5)


Machine_Learning(df_all)

#%% Network Analysis
def Net_work(df):
    df = df.sort_values(by = ["Regions","State"])
    df = df.reset_index(drop=True)
    # Set a new empty network
    G = nx.Graph()
    # Add nodes
    def add_nodes(dataframe):
        # Add counties as nodes 
        for i in range(len(dataframe)):
            G.add_node(dataframe["County_State"][i], pos = (0,i))
        # Add counties as nodes   
        min = dataframe["Insured_rate"].min()
        max = dataframe["Insured_rate"].max()
        for i in range(len(dataframe)):
            G.add_node(dataframe["Insured_rate"][i], pos = (10,(dataframe["Insured_rate"][i]- min)*1800/(max - min)))
        
        min1 = dataframe["Ave_Payment"].min()
        max1 = dataframe["Ave_Payment"].max()
        for i in range(len(dataframe)):
            G.add_node(dataframe["Ave_Payment"][i], pos = (20,(dataframe["Ave_Payment"][i]- min1)*1800/(max1 - min1)))
        
        min2 = dataframe["Population"].min()
        max2 = dataframe["Population"].max()
        for i in range(len(dataframe)):
            G.add_node(dataframe["Population"][i], pos = (30,(df_all["Population"][i]- min2)*1800/(max2 - min2)))
        
        for i in range(5):
            G.add_node(dataframe["Regions"].unique()[i], pos = (-10,400*i))
        # G.nodes(data=True)
    # Call add_nodes() function
    add_nodes(df)
    
    # Add edges
    def add_edges(dataframe):
        # Add edges between county and insured rate
        for i in range(len(dataframe)):
            G.add_edge(dataframe["County_State"][i], df_all["Insured_rate"][i],pos = i)
        # Add edges between insured rate and medical payment
        for i in range(len(dataframe)):
            G.add_edge(dataframe["Insured_rate"][i], df_all["Ave_Payment"][i],pos = i+len(dataframe))
        # Add edges between medical payment and population
        for i in range(len(dataframe)):
            G.add_edge(dataframe["Ave_Payment"][i], df_all["Population"][i],pos = i+2*len(dataframe))
        # Add edges between county and regions
        for i in range(len(dataframe)):
            G.add_edge(dataframe["County_State"][i], df_all["Regions"][i],pos = i+3*len(dataframe))
        # G.edges()
    # Call add_edges() function
    add_edges(df)
    
    # Draw the network   
    def Draw(dataframe):
        # Seperate different edges, so that we can draw them in different colours.
        edge1 = [(u,v) for (u,v,d) in G.edges(data=True) if d['pos'] < len(dataframe)]
        edge2 = [(u,v) for (u,v,d) in G.edges(data=True) if len(dataframe) <= d['pos'] < 2*len(dataframe)]
        edge3 = [(u,v) for (u,v,d) in G.edges(data=True) if 2*len(dataframe) <= d['pos'] < 3*len(dataframe)]
        edge4 = [(u,v) for (u,v,d) in G.edges(data=True) if 3*len(dataframe) <= d['pos'] < 4*len(dataframe)]
        
        # Define positions for nodes
        pos=nx.get_node_attributes(G,'pos')
        
        # Draw a network with different colours. 
        # Red nodes: county
        # Blue nodes: Insured rate
        # Purple nodes: Medical payment
        # Green nodes: Population
        # Orange nodes: Five regions
        nx.draw_networkx_nodes(G, pos,nodelist = list(dataframe["County_State"]), node_color ="red",node_size = 10)
        nx.draw_networkx_nodes(G, pos,nodelist = list(dataframe["Insured_rate"]), node_color ="blue",node_size = 80)
        nx.draw_networkx_nodes(G, pos,nodelist = list(dataframe["Ave_Payment"]), node_color ="purple",node_size = 80 )
        nx.draw_networkx_nodes(G, pos,nodelist = list(dataframe["Population"]), node_color ="green",node_size = 80 )
        nx.draw_networkx_nodes(G, pos,nodelist = list(dataframe["Regions"]), node_color ="orange",node_size = 80)
        nx.draw_networkx_edges(G,pos,edgelist = edge1 ,edge_color = "blue",alpha = 0.05)
        nx.draw_networkx_edges(G,pos,edgelist = edge2 ,edge_color = "purple",alpha = 0.1)
        nx.draw_networkx_edges(G,pos,edgelist = edge3 ,edge_color = "green",alpha = 0.05)
        nx.draw_networkx_edges(G,pos,edgelist = edge4 ,edge_color = "orange",alpha = 0.1)
        # labels
        # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
        plt.show() 
        
    # Call Draw() function
    Draw(df)  

    def local_metrics(df):
        # Prints summary information about the graph
        print(nx.info(G))
        
        # Print the degree of each node
        print("Node Degree")
        for v in G:
            print('%s %s' % (v,G.degree(v)))
        
        # Compute and print other stats    
        nbr_nodes = nx.number_of_nodes(G)
        nbr_edges = nx.number_of_edges(G)
        nbr_components = nx.number_connected_components(G)
        
        print("Number of nodes:", nbr_nodes)
        print("Number of edges:", nbr_edges)
        print("Number of connected components:", nbr_components)
        
        
        # Compute betweeness and then store the value with each node in the networkx graph
        betweenList = nx.betweenness_centrality(G)
        print();
        print("Betweeness of each node")
        print(betweenList)
        
        # Compute the clustering coefficient
        cluster_coe = nx.clustering(G)
        print();
        print("Clustering coefficient of this network")
        print(cluster_coe)
    local_metrics(df)
    
    def global_metrics(df):
        # Compute the density
        density = nx.density(G)
        print();
        print("Density of this network")
        print(density)
        
        #Compute the averages for the centrality
        ave_centrality = nx.closeness_centrality(G)
        print();
        print("Average centrality of this network")
        print(ave_centrality)
        
        # Compute the triangles 
        triangles = nx.triangles(G)
        print();
        print("Trianges of this network")
        print(triangles)
    global_metrics(df)
    
    def cluster_Anly(df):
        # Clustering
        # Conduct modularity clustering
        partition = community.best_partition(G)
        
        # Print clusters (You will get a list of each node with the cluster you are in)
        print();
        print("Clusters")
        print(partition)
        
        # Get the values for the clusters and select the node color based on the cluster value
        values = [partition.get(node) for node in G.nodes()]
        nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=100, with_labels=False)
        plt.show()
        
        # Determine the final modularity value of the network
        modValue = community.modularity(partition,G)
        print("modularity:", modValue)
    cluster_Anly(df)

Net_work(df_all)