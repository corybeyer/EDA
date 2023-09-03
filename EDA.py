#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import warnings
warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
sns.set_palette("PRGn")


# In[1]:


class Data_Explorer:
    def __init__(self, df, variable):
        self.df = df
        self.target = variable
        
        self.numeric_feature_names = self.df.select_dtypes('number').columns
        self.numeric_features = self.df[self.numeric_feature_names]
        
        self.numeric_binary_names = [var for var in self.df[self.numeric_feature_names] if len(np.unique(self.df[var])) == 2 and var != self.target]
        self.numeric_binary = self.df[self.numeric_binary_names]
        
        self.numeric_discrete_names = [var for var in self.df[self.numeric_feature_names] if len(np.unique(self.df[var])) <=20 & 
                 len(np.unique(self.df[var])) > 2 and var != self.target]
        self.numeric_discrete = self.df[self.numeric_discrete_names]
        
        self.numeric_continuous_names = [var for var in self.df[self.numeric_feature_names] if var not in self.numeric_binary_names and var not in self.numeric_discrete_names and var != self.target]
        self.numeric_continuous = self.df[self.numeric_continuous_names]
        
        self.categorical_feature_names = self.df.select_dtypes('object').columns
        self.categorical_features = self.df[self.categorical_feature_names]
                     
        self.categorical_nominal_names = [var for var in self.df[self.categorical_feature_names] if self.df[var].nunique() >= 3 and self.df[var].nunique() <= 50 and var != self.target]
        self.categorical_nominal = self.df[self.categorical_nominal_names]
        
        self.categorical_narrative_names = [var for var in self.df[self.categorical_feature_names] if self.df[var].nunique() > 51 and var != self.target]
        self.categorical_narrative = self.df[self.categorical_narrative_names]    
    
    def evaluate_nulls(self):
        null_percentages = self.df.isnull().sum()/len(self.df) * 100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_feature_nulls(self):
        null_percentages = self.df[self.numeric_feature_names].isnull().sum() / len(self.df) * 100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_binary_nulls(self):
        null_percentages = self.df[self.numeric_binary_names].isnull().sum()/len(self.df) * 100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_discrete_nulls(self):
        null_percentages = self.df[self.numeric_discrete_names].isnull().sum()/len(self.df) * 100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def categorical_feature_nulls(self):
        null_percentages = self.df[self.categorical_feature_names].isnull().sum()/len(self.df) *100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
       
    def categorical_nominal_nulls(self):
        null_percentages = self.df[categorical_nominal].isnull().sum()/len(self.df) *100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def categorical_narrative_nulls(self):
        null_percentages = self.df[self.categorical_narrative].isnull().sum()/len(self.df) *100
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_continuous_plot(self):
        num_features = len(self.numeric_continuous_names)
        num_cols = 3  # You can change this to fit your needs
        num_rows = math.ceil(num_features / num_cols)
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten the axes array to easily iterate
        
        for i, column in enumerate(self.numeric_continuous_names):
            sns.distplot(x=self.df[column], ax=axes[i])
            axes[i].set_title(f'Distribution Plot - {column}')
        
        # Remove any unused subplots
        for i in range(num_features, num_rows * num_cols):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
    def numeric_features_plot(self):
        num_features = len(self.numeric_feature_names)
        num_rows = num_features  # One row per feature
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        
        for i, column in enumerate(self.numeric_feature_names):
            sns.boxplot(data=self.df[column], ax=axes[i, 0])
            axes[i, 0].set_title(f'Boxplot - {column}')
            
            sns.histplot(data=self.df[column], ax=axes[i, 1], bins=10)
            axes[i, 1].set_title(f'Histogram - {column}')
        
        plt.tight_layout()
    
    def categorical_nominal_plot(self):
        fig, axes = plt.subplots(nrows=len(self.categorical_nominal_names), figsize=(10, 7 * len(self.categorical_nominal_names)))
        for i, column in enumerate(self.categorical_nominal_names):
            sns.countplot(data=df, x=column, ax=axes[i], order=df[column].value_counts().index)
            axes[i].set_title(f'Count Plot - {column}')
            axes[i].set_ylabel('Unique Count')
            
            for p in axes[i].patches:
                height = p.get_height()
                if pd.notna(height):
                    axes[i].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9)
                axes[i].tick_params(axis='x', labelrotation=87)
        
        plt.subplots_adjust(hspace=.5) 
    
    def numeric_features_describe(self):
        return self.numeric_features.describe().T
    
    def categorical_features_describe(self):
        return self.categorical_features.describe().T
    
    def categorical_feature_plot123(self):
        num_features = len(self.categorical_feature_names)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_features, figsize=(10, 7 * num_features))
        
        # If there's only one subplot, axes will not be an array; make it into a list for consistency
        if num_features == 1:
            axes = [axes]
        
        for i, column in enumerate(self.categorical_feature_names):
            sns.countplot(data=self.df, x=column, ax=axes[i], order=self.df[column].value_counts().index)
            axes[i].set_title(f'Count Plot - {column}')
            axes[i].set_ylabel('Unique Count')
            
            # Annotate bars with their heights
            for p in axes[i].patches:
                height = p.get_height()
                if pd.notna(height):
                    axes[i].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels if they are too long
            if max([len(str(label.get_text())) for label in axes[i].get_xticklabels()]) > 10:
                axes[i].tick_params(axis='x', labelrotation=45)
            else:
                axes[i].tick_params(axis='x', labelrotation=0)
        
        # Adjust the layout to prevent overlap
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()

