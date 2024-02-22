#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd


# In[22]:


import warnings
warnings.filterwarnings('ignore')


# In[23]:


df = pd.read_csv('customer.csv')


# In[24]:


df.head(5)


# In[25]:


df.info()


# In[26]:


df.describe(include = 'all').round(2)


# In[27]:


df.iloc[0].isna().sum()


# In[28]:


df.isnull().sum(axis=1)


# In[29]:


df.isna().sum


# In[30]:


from matplotlib import pyplot as plt


# In[31]:


import seaborn as sns
sns.set_theme()


# In[34]:


ax = sns.scatterplot(data = df, 
                     x = 'income',
                     y = 'tenure',
                     s = 50)


# In[36]:


clusterdata = df[['income', 'tenure']].describe().round(2)
clusterdata


# In[37]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[38]:


scaled_df = scaler.fit_transform(df[['income', 'tenure']])
scaled_df


# In[39]:


scaled_df = pd.DataFrame(scaled_df, columns = ['income', 'tenure'])
scaled_df


# In[40]:


scaled_df.describe().round(2)


# In[46]:


from sklearn.cluster import KMeans
k_model = KMeans(n_clusters = 2, n_init = 25, random_state = 300)
k_model.fit(scaled_df)


# In[47]:


evaluate = pd.Series(k_model.labels_).value_counts()
evaluate


# In[48]:


ax = sns.scatterplot(data = df, 
                     x = 'income',
                     y = 'tenure',
                     s = 50)


# In[56]:


centeroid = pd.DataFrame(k_model.cluster_centers_,
                        columns = ['income', 'tenure'])
centeroid


# In[57]:


plt.figure(figsize = (12, 10))

ax = sns.scatterplot(data = scaled_df,
                     x = 'income',
                     y = 'tenure',
                     hue = k_model.labels_,
                     palette = 'colorblind',
                     alpha = 0.9,
                     s = 150,
                     legend = True)
ax = sns.scatterplot(data = centeroid, 
                     x = 'income',
                     y = 'tenure',
                     hue = centeroid.index,
                     palette = 'colorblind',
                     s = 900,
                     marker = 'D',
                     ec = 'black',
                     legend = False)

for i in range(len(centeroid)):
    plt.text(x = centeroid.income[1],
             y = centeroid.tenure[1],
             s = 1,
             horizontalalignment = 'center',
             verticalalignment = 'center',
             size = 15,
             weight = 'bold',
             color = 'white')


# In[95]:


wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 300)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
wcss_s = pd.Series(wcss, index = range(1, 11))
    
plt.figure(figsize=(12, 10))
ax = sns.lineplot(y = wcss_s, x = wcss_s.index)
ax = sns.scatterplot(y = wcss_s, x = wcss_s.index, s = 200)
ax = ax.set(xlabel = 'Optimal Clusters Number (k)',
            ylabel = 'Within Cluster Sum of Squares (WCSS)')


# In[96]:


from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(scaled_df, k_model.labels_)
silhouette_score


# In[99]:


fin_model = KMeans(n_clusters = 3, n_init = 25, random_state = 300)
fin_model.fit(scaled_df)


# In[101]:


centeroid = pd.DataFrame(fin_model.cluster_centers_,
                         columns = ['income', 'tenure'])
centeroid


# In[102]:


plt.figure(figsize=(12, 10))

ax = sns.scatterplot(data = scaled_df,
                     x = 'income',
                     y = 'tenure',
                     hue = fin_model.labels_,
                     palette = 'colorblind',
                     alpha = 0.9,
                     s = 200,
                     legend = True)

ax = sns.scatterplot(data = centeroid,
                     x = 'income',
                     y = 'tenure',
                     hue = centeroid.index,
                     palette = 'colorblind',
                     s = 900,
                     marker = 'D',
                     ec = 'black',
                     legend = False)

for i in range(len(centeroid)):
    plt.text(x = centeroid.income[i],
             y = centeroid.tenure[i],
             s = i,
             horizontalalignment = 'center',
             verticalalignment = 'center',
             size = 15,
             weight = 'bold',
             color = 'white')


# In[105]:


df['cluster'] = fin_model.labels_.tolist()
df.head(12)


# In[106]:


customers = pd.get_dummies(df, columns = ['gender'])
customers.head(12)


# In[123]:


analysis = customers.agg((
    gender_Female := 'mean',
    gender_Male := 'mean',
    age := 'median',
    income := 'median',
    tenure := 'median')).round(2)
analysis


# In[124]:


customers.groupby('cluster').agg((
    gender_Female := 'mean',
    gender_Male := 'mean',
    age := 'median',
    income := 'median',
    tenure := 'median'))


# In[ ]:




