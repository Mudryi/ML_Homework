
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[9]:


def drop_nan(data, colum =None, axis=0, inplace = False):
    if inplace:
        df = data
    else:
        df = data.copy()
    
    if axis == 0:
        if colum is not None:
            indexes = df[df[colum].isnull()].index.values.astype(int)
            df = df.drop(list(indexes))
        else:
            for j in df:
                try:
                    indexes = df[df[j].isnull()].index.values.astype(int)
                    df = df.drop(list(indexes))
                except:
                    continue
    
    if axis == 1:
        if colum is not None:
            if df[colum].isnull().sum() > 0:
                df = df.drop([colum],axis=1)
        else:
            df = df.drop(df.columns[df.isna().any()].tolist(),axis=1)
    return df.reset_index()


# In[2]:


def repl_nan(data,colum= None, inplace = True, type='mean'):
    if inplace:
        fr = data
    else:
        fr = data.copy()
    if colum is None:
        for j in fr:
            try:
                if type == 'mean':
                    fr[j] = fr[j].astype(float)
                for i in range(len(fr.isnull())):
                    if fr[j].isnull()[i] == True:
                        if type == 'mean':
                            fr[j][i] = fr[j].mean()
                        if type == 'median':
                            fr[j][i] = fr[j].median()
                        if type == 'mode':
                            fr[j][i] = fr[j].mode()[0]
            except:
                continue
    else:
        j = colum
        try:
            if type == 'mean':
                fr[j] = fr[j].astype(float)
            for i in range(len(fr[j])):
                if fr[j].isnull()[i] == True:
                    if type == 'mean':
                        fr[j][i] = fr[j].mean()
                    if type == 'median':
                        fr[j][i] = fr[j].median()
                    if type == 'mode':
                        fr[j][i] = fr[j].mode()[0]
        except:
            print('Err')
    return fr


# In[3]:


def rep_nan_reg(data,to_replace,base_on,inplace = False):
    if inplace:
        df = data
    else:
        df = data.copy()
    x_train = df[df[to_replace].notnull() & df[base_on].notnull()][base_on]
    y_train = df[df[to_replace].notnull() & df[base_on].notnull()][to_replace]
    x = x_train.as_matrix().reshape(-1,1)
    y = y_train.as_matrix().reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    y_test = df[df[to_replace].isnull() & df[base_on].notnull()][base_on]
    y_test = y_test.as_matrix().reshape(-1,1)
    a = reg.predict(y_test)
    k = 0
    for i in range(len(df[to_replace].isnull())):
        if df[to_replace].isnull()[i] == True:
            df[to_replace][i] = a[k][0]
            k+=1   
    return df


# In[4]:


def standardize(df,colum = None,inplace = False):
    if inplace:
        data = df
    else:
        data = df.copy()
    if colum is None:    
        for j in data:
            try:
                data[j] = data[j].apply(lambda x: (x - data[j].mean())/data[j].std())
            except:
                continue
    else:
        j = colum
        try:
            data[j] = data[j].apply(lambda x: (x - data[j].mean())/data[j].std())
        except:
            print('Error')
    return data


# In[5]:


def normalize(df,colum = None,inplace = False):
    if inplace:
        data = df
    else:
        data = df.copy()
    if colum is None:  
        for j in data:
            try:
                data[j] = data[j].astype(float)
                data[j] = data[j].apply(lambda x: (x - data[j].min())/(data[j].max()-data[j].min()))
            except:
                continue
    else:
        j = colum
        try:
            data[j] = data[j].astype(float)
            data[j] = data[j].apply(lambda x: (x - data[j].min())/(data[j].max()-data[j].min()))
        except:
            print('Err')
    return data


# In[6]:


def euclidian_dist(x_known,x_unknown):
    
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]

    dists = np.empty((num_pred,num_data))

    for i in range(num_pred):
        for j in range(num_data):
            dists[i,j] = np.sqrt(np.sum((x_unknown[i]-x_known[j])**2))
            
    return dists


def k_nearest_labels(dists, y_known, k):

    num_pred = dists.shape[0]
    n_nearest = []
    
    for j in range(num_pred):
        dst = dists[j]

        closest_y = [y_known[i] for i in np.argsort(dst)[:k]]
        
        n_nearest.append(closest_y)
    return np.asarray(n_nearest) 


# In[8]:


def KNN(data,base_on,predict,count_neib = 3,k=3,inplace = False):
    if inplace:
        df = data
    else:
        df = data.copy()
    indexes = df[df[predict].isnull()].index.values.astype(int)
    train = []
    test = []
    tr_labels = []
    for i in indexes:
        test.append(list(df[base_on].iloc[i].astype(float)))
        for j in range(1,count_neib+1):
            try:
                train.append(list(df[base_on][df[predict].notnull()].iloc[i+j].astype(float)))
                tr_labels.append(df[predict][df[predict].notnull()].iloc[i+j])
                train.append(list(df[base_on][df[predict].notnull()].iloc[i-j].astype(float)))
                tr_labels.append(df[predict][df[predict].notnull()].iloc[i-j])
            except:
                continue
    train = np.array(train)
    test = np.array(test)
    tr_labels = np.array(tr_labels)
    d = euclidian_dist(train,test)
    des = k_nearest_labels(d,tr_labels,k)
    labels = []
    for q in range(0,des.shape[0]):
        a, b = np.unique(des[q], return_counts=True)
        labels.append(a[b.argmax()])
    k = 0
    for i in indexes:
        df[predict] = labels[k]
        k +=1 
    return df

