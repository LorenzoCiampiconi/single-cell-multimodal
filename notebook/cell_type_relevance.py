#!/usr/bin/env python
# coding: utf-8

#% In[1]:


import pandas as pd
import pathlib
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


DATA = pathlib.Path("../data")


# In[3]:


metadata = pd.read_csv("/Users/lciampiconi/PycharmProjects/kaggle/single-cell-multimodal/metadata.csv")


# In[4]:


metadata.head(20)


# In[5]:


metadata.cell_type.value_counts().plot.bar()


# In[6]:


metadata_cite = metadata[metadata.technology == "citeseq"]
metadata_cite.cell_type.value_counts().plot.bar()


# In[7]:


target_cite = pd.read_hdf(DATA / "train_cite_targets.h5")


# In[8]:


target_cite.head(20)


# In[9]:


target_cite_w_cell_type = target_cite.reset_index().merge(metadata_cite[["cell_type", "cell_id"]], on="cell_id")


# In[10]:


target_cite.index


# In[11]:


target_cite_w_cell_type.head(20)


# In[12]:


target_cite_w_cell_type.set_index("cell_id", inplace=True)


# In[45]:


sns.displot(target_cite_w_cell_type, x="CD62L", hue="cell_type", kind="kde", height=5, aspect=2.5)


# In[43]:


for c in target_cite_w_cell_type.columns[:-1]:
    print(c)
    df = target_cite_w_cell_type[[c, "cell_type"]]
    sns.displot(df, x=c, hue="cell_type", kind="kde", height=5, aspect=2.5, clip=(-4, 4))


# In[ ]:
