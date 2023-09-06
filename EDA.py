#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#import textwrap
import warnings
warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
sns.set_palette("PRGn")

# In[1]:
class Data_Explorer:
    def __init__(self, df, target):
        self.df = df.drop(columns = target).copy()
        self.target = df[target]
        
        """select boolean features"""
        self.boolean_names = self.df.select_dtypes(include='boolean').columns
        self.boolean_features = self.df[self.boolean_names]

        """select numeric features"""
        self.numeric_names = self.df.select_dtypes('number').columns
        self.numeric_features = self.df[self.numeric_names]

        """select discrete features"""
        self.discrete_names = [var for var in self.df[self.numeric_names] if len(np.unique(self.df[var])) <=20]
        self.discrete_features = self.df[self.discrete_names]

        self.categorical_names = self.df.select_dtypes('object').columns
        self.categorical_features = self.df[self.categorical_names]

        self.dates_names = self.df.select_dtypes('datetime').columns
        self.dates_features = self.df[self.dates_names]

    def target_nulls(self):
        null_percentages = round(self.target.isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
        ## error required next line needed [null_percentages] instead of null_percentages w/o bracekts becuase it prevoiusly returned 0.00 and not a series
        temp_df = pd.DataFrame([null_percentages], columns = column)
        return temp_df
    
    def target_plot(self):
        if self.target.dtype == 'O':
            fig, axe = plt.subplots()
            axe.set_title(f'Count Plot - {self.target.name}')
            sns.countplot(x = self.target, ax = axe)
        else:
            fig, axe = plt.subplots()
            axe.set_title(f'Histogram Plot - {self.target.name}')
            sns.histplot(x = self.target, ax = axe)

    def target_class_balance_binary(self):
        total_rows = len(self.df)
        counts = self.target.value_counts()

        data_check = set(counts.index)
        if data_check != {0,1}:
            return "This is not binary data"
        
        ones = counts.get(1,0)
        zeros = counts.get(0,0)
        if zeros != 0:
            ratio_ones_to_zeros = ones/zeros
        else:
            ratio_ones_to_zeros = "Undefined, division by zero error"

        ratio_ones_to_all = ones/total_rows
        
        data = {
        "Ratio of 1s to 0s": ratio_ones_to_zeros,
        "Ratio of 1s to all entries": ratio_ones_to_all
        }
        
        return pd.DataFrame(data)    
    
    def boolean_nulls(self):
        if self.boolean_names.empty:
            return "No boolean columns selected."
        
        null_percentages = round(self.df[self.boolean_names].isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def boolean_plot(self):
        if self.boolean_names.empty:
            return "No boolean columns selected. Cannot plot null values."
        
        fig, axes = plt.subplots(nrows=len(self.boolean_names), figsize=(10, 7 * len(self.boolean_names)))
        for i, column in enumerate(self.boolean_names):
            sns.countplot(data=self.df, x=column, ax=axes[i], order=self.df[column].value_counts().index)
            axes[i].set_title(f'Count Plot - {column}')
            axes[i].set_ylabel('Unique Count')
            
            for p in axes[i].patches:
                height = p.get_height()
                if pd.notna(height):
                    axes[i].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9)
                axes[i].tick_params(axis='x', labelrotation=87)
        
        plt.subplots_adjust(hspace=.5) 

    def discrete_nulls(self):
        if not self.discrete_names:
            return "No discrete columns selected. Cannot plot null values."
        null_percentages = round(self.df[self.discrete_names].isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def discrete_plot(self):
        # Calculate the number of rows needed
        nrows = len(self.discrete_names)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows))
        # If there's only one row, axes is a 1D array
        if nrows == 1:
            axes = np.reshape(axes, (1, -1))
        # Loop through each column name
        for i, column in enumerate(self.discrete_names):
            sns.boxplot(data=self.df, x=column, ax=axes[i, 0])
            axes[i, 0].set_title(f'Boxplot - {column}')
            
            sns.histplot(data=self.df, x=column, ax=axes[i, 1], bins=10)
            axes[i, 1].set_title(f'Histogram - {column}')
            
            axes[i, 0].tick_params(axis='x', labelrotation=45)
            axes[i, 1].tick_params(axis='x', labelrotation=45)

        plt.tight_layout()
        plt.show()
    
    """Displays percentages of null values per numeric column name"""
    def numeric_nulls(self):
        if self.numeric_names.empty:
            return "No boolean columns selected. Cannot plot null values."
        null_percentages = round(self.df[self.numeric_names].isnull().sum() / len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df

    def numeric_plot(self):
        num_features = len(self.numeric_names)
        num_rows = num_features  # One row per feature       
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        for i, column in enumerate(self.numeric_names):
            sns.boxplot(data=self.df[column], ax=axes[i, 0])
            axes[i, 0].set_title(f'Boxplot - {column}')
            
            sns.histplot(data=self.df[column], ax=axes[i, 1], bins=10)
            axes[i, 1].set_title(f'Histogram - {column}')
        plt.tight_layout()

    def categorical_nulls(self):
        null_percentages = round(self.df[self.categorical_names].isnull().sum()/len(self.df) *100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df

    def categorical_feature_plot(self, variables =  None):
        if not isinstance(variables, list):
            raise ValueError("The variables input must be a list.")
        else:
            if variables == None:
                num_features = len(self.categorical_names)
        
                # Create a figure with subplots
                fig, axes = plt.subplots(nrows=num_features, figsize=(10, 7 * num_features))
        
                # If there's only one subplot, axes will not be an array; make it into a list for consistency
                if num_features == 1:
                    axes = [axes]
            
                for i, column in enumerate(self.categorical_names):
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
            
            else:
                num_features = len(self.categorical_names.drop(variables))
                # Create a figure with subplots
                fig, axes = plt.subplots(nrows=num_features, figsize=(10, 7 * num_features))
        
                # If there's only one subplot, axes will not be an array; make it into a list for consistency
                if num_features == 1:
                    axes = [axes]
            
                for i, column in enumerate(self.categorical_names.drop(variables)):
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
    def numeric_describe(self):
        return self.numeric_features.describe().T

    def target_describe(self):
        return self.target.describe().T
    
    def categorical_describe(self):
        return self.categorical_features.describe().T
    
    def categorical_to_target(self, variables=None):
        if variables is None:
            num_rows = len(self.categorical_names)
            fig, axe = plt.subplots(nrows=num_rows, figsize=(10, 7 * num_rows))
            for i, column in enumerate(self.categorical_names):
                sns.countplot(data=self.categorical_features, x=column, hue = self.target, ax=axe[i])
        else:
            if not isinstance(variables, list):
                raise ValueError("The variables input must be a list if provided.")
            else:
                num_rows = len(self.categorical_names.drop(variables))
                fig, axe = plt.subplots(nrows=num_rows, figsize=(10, 7 * num_rows))
                for i, column in enumerate(self.categorical_names.drop(variables)):
                    sns.countplot(data=self.categorical_features, x=column, hue = self.target, ax=axe[i])
