# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:19:44 2020

@author: lcr
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#-----------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------
def PreCleaning_Dataframe(df):
    # Sample
    print('\nSample of dataframe:\n{}'.format(df.head(5)))
    # Dimensions
    n_rows=df.shape[0]
    print('\nDimensions of dataframe:\n{}'.format(df.shape))
    # Data Type in Columns
    print('\nData Type and Missing Data in Columns')
    print(df.info())
    # Unique Values Columns
    print('\nUnique Values Columns')
    i=0
    for col in df:
        if n_rows==df[col].nunique():
            print(col)
            i+=1
            pass
        pass
    if i==0:
        print('none')
        pass
    pass

def VbleType_Dataframe(df,df_vbletypes):
    result={}
    k=0
    for i in df:
        result[i]=df_vbletypes[k]
        k+=1
        pass
    return result

def MissingData_Dataframe(df,df_vbletype):
    n_missing=0
    df_missing=pd.DataFrame()
    # Detection of Missing Values
    for i,i_type in df_vbletype.items():
        if i_type=='numerical_discrete' or i_type=='numerical_continuous':
            n_mis,df_mis=MissingData_Dataframe_Numerical(df,i)
            pass
        if i_type=='categorical_nominal' or i_type=='categorical_ordinal':
            n_mis,df_mis=MissingData_Dataframe_Categorical(df,i)
            pass
        if i_type=='time_serie':
            pass
        n_missing+=n_mis
        df_missing=pd.concat([df_missing,df_mis]).drop_duplicates()
        pass
    # Missingness
    if n_missing>0:
        # msno.bar(df)
        # msno.matrix(df)
        # msno.heatmap(df)
        # msno.dendrogram(df)
        pass
    return df_missing
def MissingData_Dataframe_Numerical(df,vble):
    # Detection of Missing Values
    df_missing=df[df[vble].isnull()]
    n_missing=df_missing.shape[0]
    if n_missing>0:
        print('\n{} missing values in {}'.format(n_missing,vble))
        pass
    return n_missing,df_missing
def MissingData_Dataframe_Categorical(df,vble):
    # Detection of Missing Values
    df_missing=df[df[vble].isnull()]
    n_missing=df_missing.shape[0]
    if n_missing>0:
        print('\n{} missing values in {}'.format(n_missing,vble))
        pass
    return n_missing,df_missing
def MissingData_Delete_Dataframe(df,missing):
    df_NOmissing=df
    for index,row in missing.iterrows():
        print(index)
        df_NOmissing=df_NOmissing.drop(index=index)
        pass
    return df_NOmissing

def DataConversion_Dataframe(df,df_vbletype):
    df_converted=df
    df_conversions=pd.DataFrame(columns=['vble','value','meaning'])
    df_conversions_aux=pd.DataFrame()
    for i,i_type in df_vbletype.items():
        print(i)
        if i_type=='numerical_discrete':
            pass
        if i_type=='numerical_continuous':
            df_converted=DataConversion_Dataframe_NumericalContinuous(df,i)
            pass
        if i_type=='categorical_nominal':
            df_converted,df_conversions_aux=DataConversion_Dataframe_CategoricalNominal(df_converted,i)
            df_conversions=pd.concat([df_conversions,df_conversions_aux])
            pass
        if i_type=='categorical_ordinal':
            df_converted,df_conversions_aux=DataConversion_Dataframe_CategoricalOrdinal(df_converted,i)
            df_conversions=pd.concat([df_conversions,df_conversions_aux])
            pass
        if i_type=='time_serie':
            pass
        pass
    # Missingness
    return df_converted,df_conversions
def DataConversion_Dataframe_NumericalContinuous(df,vble):
    df_converted=df
    df_converted[vble]=df_converted[vble].astype(float)
    return df_converted
def DataConversion_Dataframe_CategoricalNominal(df,vble):
    df_converted=df
    df_conversions=pd.DataFrame(columns=['vble','value','meaning'])
    n_values=df[vble].nunique()
    if n_values==1:
        print('\n{} is a constant, not a variable'.format(vble))
        pass
    if n_values==2:
        print('\n{} is a binary variable'.format(vble))
        first_value,second_value=sorted(df[vble].unique().tolist(),reverse=True)
        df_converted[vble]=df_converted[vble].apply(lambda x: 1 if x==first_value else 0)
        df_aux=pd.DataFrame([[vble,1,first_value]],columns=['vble','value','meaning'])
        df_conversions=pd.concat([df_conversions,df_aux])
        df_aux['value']=0
        df_aux['meaning']=second_value
        df_conversions=pd.concat([df_conversions,df_aux])
        pass
    else:
        le=LabelEncoder()
        df[vble]=le.fit_transform(df[vble])
        df_aux=pd.DataFrame([[vble,1,0]],columns=['vble','value','meaning'])
        for encode in le.classes_:
            df_aux['value']=encode
            df_aux['meaning']=le.transform([encode])
            df_conversions=pd.concat([df_conversions,df_aux])
            pass
        pass
    return df_converted,df_conversions
def DataConversion_Dataframe_CategoricalOrdinal(df,vble):
    df_converted=df
    df_conversions=pd.DataFrame(columns=['vble','value','meaning'])
    n_values=df[vble].nunique()
    if n_values==1:
        print('\n{} is a constant, not a variable'.format(vble))
        pass
    if n_values==2:
        print('\n{} is a binary variable'.format(vble))
        first_value,second_value=sorted(df[vble].unique().tolist(),reverse=True)
        df_converted[vble]=df_converted[vble].apply(lambda x: 1 if x==first_value else 0)
        df_aux=pd.DataFrame([[vble,1,first_value]],columns=['vble','value','meaning'])
        df_conversions=pd.concat([df_conversions,df_aux])
        df_aux['value']=0
        df_aux['meaning']=second_value
        df_conversions=pd.concat([df_conversions,df_aux])
        pass
    return df_converted,df_conversions
def DataConversion_Dataframe_FromString_ToDatetimeYear(df,vble):
    df_converted=df
    df_converted[vble]=pd.to_datetime(df_converted[vble])
    df_converted[vble]=pd.DatetimeIndex(df_converted[vble]).year
    return df_converted
def DataConversion_Dataframe_FromTimestamp_ToDatetimeYear(df,vble):
    df_converted=df
    df_converted[vble]=pd.to_datetime(df_converted[vble],unit='s')
    df_converted[vble]=pd.DatetimeIndex(df_converted[vble]).year
    return df_converted

def UniquenessTest(df,vble):
    # Dimensions
    n_rows=df.shape[0]
    if n_rows==df[vble].nunique():
        print('column has unique values')
    else:
        # Value_Counts
        dict_valcounts=df[vble].value_counts().to_dict()
        repeated_list = list({k:v for k,v in dict_valcounts.items() if v>1}.keys())
        # Unique Percentage
        repeated_items=len(repeated_list)
        print('\nUnique Percentage:\n{} ({}/{})'.format((n_rows-repeated_items)*100/n_rows,(n_rows-repeated_items),n_rows))
        # Repeated Percentage
        print('\nRepeated Percentage:\n{} ({}/{})'.format(repeated_items*100/n_rows,repeated_items,n_rows))
        # Repeated items
        print('\nRepeated Items:\n')
        for i in repeated_list:
            print(i)
            pass
        return repeated_list
    pass
def UniqueIndex(df,vble,repeated_vble,unique_index=None,check_similarity=True):
    if unique_index==None:
        unique_index==pd.DataFrame(columns=['repeated','single'])
        pass
    unique_index_aux=pd.DataFrame(columns=['repeated','single'])
    for i in repeated_vble:
        single,list_repeated=LeftOneOfRepeated(df,vble,i,check_similarity=check_similarity)
        for j in list_repeated:
            df=df.drop(j)
            unique_index_aux=pd.DataFrame([[j,single]],columns=['repeated','single'])
            unique_index=pd.concat([unique_index,unique_index_aux])
            pass
        pass
    return df,unique_index
def LeftOneOfRepeated(df,vble,val,check_similarity=True):
    repeated=df[df[vble]==val]
    if repeated.shape[0]>1:
        index_left=repeated.index[0]
        index_substitution=[]
        similarity=True
        if check_similarity==True:
            # Check Similarity
            for column in df:
                if repeated[column].nunique()!=1:
                    similarity=False
                    print()
                    pass
                pass
            pass
        if similarity==True:
            for i in range(1,repeated.shape[0]):
                index_substitution.append(repeated.index[i])
                pass
            pass
        return index_left,index_substitution
    pass

def EDA_Univariate_Dataframe(df,df_vbletype):
    # Categorical
    print('\nCategorical Variables')
    for i,i_type in df_vbletype.items():
        if i_type=='categorical_nominal' or i_type=='categorical_ordinal':
            EDA_Univariate_Dataframe_Categorical(df,i)
            pass
        pass
    # Numerical
    print('\nNumerical Variables')
    for i,i_type in df_vbletype.items():
        if i_type=='numerical_discrete' or i_type=='numerical_continuous':
            EDA_Univariate_Dataframe_Numerical(df,i)
            pass
        pass
def EDA_Univariate_Dataframe_Categorical(df,vble):
    print('\n{} has {} different values'.format(vble,df[vble].nunique()))
    if df[vble].nunique()<=25:
        # Plot Bar
        print('\n{} plotted'.format(vble))
        fig=plt.figure()
        df[vble].value_counts().sort_index().plot(kind = 'bar',title=vble)
        fig.show()
        pass
    else:
        # NO Plot
        print('\n{} NOT plotted'.format(vble))
        print(print('\n{} has more than 25 different values ({})'.format(vble,df[vble].nunique())))
        pass
    pass
def EDA_Univariate_Dataframe_Numerical(df,vble):
    if df[vble].nunique()<=25:
        # Plot Histogram
        print('\n{} has {} different values'.format(vble,df[vble].nunique()))
        print('\n{} plotted'.format(vble))
        fig=plt.figure()
        df[vble].value_counts().sort_index().plot(kind = 'bar',title=vble)
        fig.show()
    else:
        # Plot CDF
        print('\n{} statistical summary'.format(vble))
        print(df[vble].describe())
        print('\n{} plotted'.format(vble))
        x,y=cdf(df[vble])
        fig=plt.figure()
        plt.scatter(x,y,marker='.',s=0.5)
        plt.title(vble)
        fig.show()
    pass

def EDA_Multivariate_Dataframe(df,hue=None):
    fig=plt.figure()
    # sns.pairplot(df,hue=hue,diag_kind='hist')
    sns.heatmap(df.corr(),annot=True,fmt='.2f')
    fig.show()
    pass

def cdf(data):
    """Compute cdf for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the cdf: x
    x = np.sort(data)
    # y-data for the cdf: y
    y = np.arange(1, len(x)+1)/n
    return x,y

def MoviesViews(movies,movies_ratings,genre):
    # Select Genre
    if genre!='All':
        movies=movies[movies[genre]==1]
        pass
    movies_id_genre=movies['movie_id'].tolist()
    # Movies Views
    movies_views=(~movies_ratings[movies_id_genre].isnull()).sum()
    return movies_views

def MoviesPopularity(movies,movies_ratings,genre):
    # Select Genre
    if genre!='All':
        movies=movies[movies[genre]==1]
        pass
    movies_id_genre=movies['movie_id'].tolist()
    # Movies Popularity=mean of ratings
    movies_popularity=movies_ratings[movies_id_genre].mean()
    return movies_popularity


