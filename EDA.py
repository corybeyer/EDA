import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import math
from scipy.stats import ttest_ind, t
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans,DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from matplotlib.colors import to_rgb, to_hex
sns.set_style("darkgrid")
sns.set_palette("PRGn")

def darken_color(color, factor=0.8):
    rgb = to_rgb(color)
    darker_rgb = [x * factor for x in rgb]
    return to_hex(darker_rgb)

# Create the original palette
original_palette = sns.color_palette("PRGn")

# Darken the colors in the palette
darker_palette = [darken_color(color) for color in original_palette]

# Set the darker palette
sns.set_palette(darker_palette)

# Your custom colors
my_colors = ['#6a3976', '#977ea2', '#bcb1bd', '#b4c2b1', '#7caa79', '#2a723a']

# Create a custom colormap
my_cmap = LinearSegmentedColormap.from_list('my_cmap', my_colors)

def custom_log(x, small_value=1e-10):
    if x <= 0:
        x = small_value
    return math.log(x) 

class DataExplorer:
    def __init__(self, df, target):
        
        # making a copy to ensure edits only exist in a save version
        self.df = df.drop(columns = target).copy()
        
        # Making a target object based on the input parameter
        self.target = df[target]
        
        """numeric columns"""       
        # A selection taken at the start to isolate numeric values in the dataframe. The numeric columns will be separated into subsets.
        get_numeric = self.df.select_dtypes('number').columns
        
        """numeric discrete features"""
        # Finding numeric values that have less than or equal to 100 distinct values. Meant to show which columns could be
        # discrete values like 1 = Good, 2 = Bad, 3 = Ugly
        self.numeric_discrete_names = [var for var in self.df[get_numeric] if len(np.unique(self.df[var])) <= 40]
        
        # find the names of the discrete column names
        self.numeric_discrete_features = self.df[self.numeric_discrete_names]

        """numeric continuous features"""     
        # The anti-selection for numeric values, which produces only continuous numeric values
        self.numeric_continuous_features = self.df[get_numeric].drop(columns=self.numeric_discrete_names)
        
        # Finding the names of the continuous columns
        self.numeric_continuous_names = self.numeric_continuous_features.columns
        
        """numeric boolean features"""
        # Finding boolean columns, a subset of numeric columns
        self.numeric_boolean_names = self.df.select_dtypes(include='boolean').columns
        
        # Getting the boolean column names 
        self.numeric_boolean_features = self.df[self.numeric_boolean_names]        
        
        """categorical columns"""              
        # A placeholder for all categorical data. Subsets will be taken from this group
        get_categorical = self.df.select_dtypes('object').columns
        
        
        """binary category features"""
        self.categorical_binary_names = [var for var in get_categorical if len(np.unique(self.df[var].astype(str))) == 2]      
        self.categorical_binary_features = self.df[self.categorical_binary_names]
        
        
        """narratives features"""
        self.categorical_narrative_names = [var for var in get_categorical if len(np.unique(self.df[var].astype(str))) >= 100]
        self.categorical_narrative_features = self.df[self.categorical_narrative_names]
        
        
        """nominal features"""
        self.categorical_nominal_names = [var for var in get_categorical if len(np.unique(self.df[var].astype(str))) < 100 and var not in self.categorical_binary_names]
        self.categorical_nominal_features = self.df[self.categorical_nominal_names]
      
        # Identify categorical features excluding binary and narrative features
        self.categorical_features = self.df.drop(columns=self.categorical_binary_names + self.categorical_narrative_names + self.categorical_nominal_names)
        
        # Get column names
        self.categorical_names = self.categorical_features.columns
        
        
        self.dates_names = self.df.select_dtypes('datetime').columns
        self.dates_features = self.df[self.dates_names]
        
    """All Things Target """
        
    def target_describe(self):
        return self.target.describe().T
        
    def target_nulls(self):
        null_percentages = round(self.target.isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
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
            sns.histplot(self.target.apply(lambda x: custom_log(x)), ax = axe)
        
    def target_class_balance_binary(self):
        total_rows = len(self.df)
        counts = self.target.value_counts()
        data_check = set(counts.index)
        # checking for binary data
        if data_check != {0,1}:
            return "The Target is not binary data, therefore balance is not calculated"
        ones = counts.get(1,0)
        zeros = counts.get(0,0)
        if zeros != 0:
            ratio_ones_to_zeros = ones/zeros
        else:
            ratio_ones_to_zeros = "Undefined, division by zero error"

        ratio_ones_to_all = ones/total_rows
        data = {
           "Type": ["Ratio of 1s to 0s","Ratio of 1s to all entries"],
            "Values": [ratio_ones_to_zeros, ratio_ones_to_all]
            }
        return pd.DataFrame(data) 
        
    """All Things Numeric Continuous"""
    
    def numeric_continuous_describe(self):
        return self.numeric_continuous_features.describe().T
    
    def numeric_continuous_nulls(self):
        if self.numeric_continuous_names.empty:
            return "No continuous columns available. Cannot plot null values."
        null_percentages = round(self.df[self.numeric_continuous_names].isnull().sum() / len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_continuous_plot(self, log=False):
        num_features = len(self.numeric_continuous_names)
        num_rows = num_features  # One row per feature       
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        for i, column in enumerate(self.numeric_continuous_names):
            if log:
                # Apply log transformation
                sns.boxplot(data=self.df[column].apply(lambda x: custom_log(x)), ax=axes[i, 0])
                sns.histplot(data=self.df[column].apply(lambda x: custom_log(x)), ax=axes[i, 1], bins=10)
            else:
                # Plot without log transformation
                sns.boxplot(data=self.df[column], ax=axes[i, 0])
                sns.histplot(data=self.df[column], ax=axes[i, 1], bins=10)
            axes[i, 0].set_title(f'Boxplot - {column}')
            axes[i, 1].set_title(f'Histogram - {column}')
        plt.tight_layout()
        
    def numeric_continuous_to_target(self, variables=None):
        is_binary_target = len(np.unique(self.target)) == 2
        df_ = self.numeric_continuous_features.copy().applymap(lambda x: custom_log(x))
        if variables is not None:
            if not isinstance(variables, list):
                raise ValueError("The variables input must be a list if provided.")
            df_.drop(columns=variables, inplace=True)
        df_['Target'] = self.target.apply(lambda x: custom_log(x) if not is_binary_target else x)
        if is_binary_target:
            t_tests = {}
            cohens_d = {}
            for col in df_.columns[:-1]:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.boxplot(x='Target', y=col, data=df_)
                plt.title(f'Boxplot of {col}')
                plt.subplot(1, 2, 2)
                sns.swarmplot(x='Target', y=col, data=df_)
                plt.title(f'Swarmplot of {col}')
                plt.tight_layout()
                plt.show()
                group1 = df_[df_['Target'] == 0][col]
                group2 = df_[df_['Target'] == 1][col]
                t_stat, p_val = ttest_ind(group1, group2)
                pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)
                d = (group1.mean() - group2.mean()) / pooled_std
                t_tests[col] = p_val
                cohens_d[col] = d
            # Print the test results as a table
            test_results_df = pd.DataFrame({
                'T-Test p-value': t_tests,
                "Cohen's d": cohens_d})
            print("Test Results Table:")
            print(test_results_df)
            # Printout text describing the test results and how to interpret them
            print("\nInterpretation:")
            print("1. The null hypothesis posits that the binary target variable has no effect on the continuous variable being tested.T-Test p-value: A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.")
            print("2. Cohen's d measures how big the difference is between the two groups. Cohen's d: A d=0.2 be considered a 'small' effect size, 0.5 represents a 'medium' effect size and 0.8 a 'large' effect size.")
        else:
            g = sns.pairplot(df_, diag_kind='kde')
            g.map_lower(sns.kdeplot, levels=4, color=".05")
            pearson_corr = df_.corr(method='pearson')
            covariance = df_.cov()
            spearman_corr = df_.corr(method='spearman')
            # Create a DataFrame to hold the results
            results_df = pd.DataFrame({
                'Pearson Correlation': pearson_corr['Target'],
                'Covariance': covariance['Target'],
                'Spearman Rank Correlation': spearman_corr['Target']})
            # Drop the 'Target' row as it will always be 1 for all measures
            results_df.drop('Target', inplace=True)
            print("Statistical Measures Table:")
            print(results_df)
            print("\nInterpretation:")
            print("1. Pearson Correlation: Measures the linear relationship between variables. Ranges from -1 (negative linear relationship) to 1 (positive linear relationship).")
            print("2. Covariance: Indicates the direction of the linear relationship between variables. Positive values signify that the variables move in the same direction, while negative values signify that they move in opposite directions.")
            print("3. Spearman Rank Correlation: Measures the strength and direction of the monotonic relationship between variables. Useful for non-linear or non-normally distributed data. Ranges from -1 to 1.")

        
    """All Things Numeric Boolean"""        
    
    def numeric_boolean_describe(self):
        if len(self.numeric_boolean_names) == 0:
            print("No data available for comparing.")
            return
        return self.numeric_boolean_features.describe().T
    
    def numeric_boolean_nulls(self):
        if self.numeric_boolean_names.empty:
            return "No boolean columns available."
        null_percentages = round(self.df[self.numeric_boolean_names].isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_boolean_plot(self):
        if self.numeric_boolean_names.empty:
            return "No boolean columns selected. Cannot plot null values."
        fig, axes = plt.subplots(nrows=len(self.numeric_boolean_names), figsize=(10, 7 * len(self.numeric_boolean_names)))
        for i, column in enumerate(self.numeric_boolean_names):
            sns.countplot(data=self.df, x=column, ax=axes[i], order=self.df[column].value_counts().index)
            axes[i].set_title(f'Count Plot - {column}')
            axes[i].set_ylabel('Unique Count')
            for p in axes[i].patches:
                height = p.get_height()
                if pd.notna(height):
                    axes[i].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9)
                axes[i].tick_params(axis='x', labelrotation=87)
        plt.subplots_adjust(hspace=.5)         

    def numeric_boolean_to_target(self):
        is_continuous_target = self.target.dtype in ['float64', 'int64']
        for col in self.numeric_boolean_features.columns:
            if is_continuous_target:
                plt.figure(figsize=(10, 6))
                # Histogram with hues
                sns.histplot(data=self.df, x=self.target, hue=col, element="step", stat="density", common_norm=False)
                plt.title(f'Distribution of Continuous Target vs {col}')
                plt.show()
                # KDE Plot with hues
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=self.df, x=self.target, hue=col)
                plt.title(f'KDE Plot of Continuous Target vs {col}')
                plt.show()           
    
    """All Things Numeric Discrete"""        
    
    def numeric_discrete_describe(self):
        if len(self.numeric_discrete_names) == 0:
            print("No data available for comparing.")
            return
        return self.numeric_discrete_features.describe().T
    
    def numeric_discrete_nulls(self):
        if not self.numeric_discrete_names:
            return "No discrete columns selected. Cannot plot null values."
        null_percentages = round(self.df[self.numeric_discrete_names].isnull().sum()/len(self.df) * 100,3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns = column)
        return temp_df
    
    def numeric_discrete_plot(self):
        if len(self.numeric_discrete_names) == 0:
            print("No data available for plotting.")
            return
        # Calculate the number of rows needed
        nrows = len(self.numeric_discrete_names)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows))
        # If there's only one row, axes is a 1D array
        if nrows == 1:
            axes = np.reshape(axes, (1, -1))
        # Loop through each column name
        for i, column in enumerate(self.numeric_discrete_names):
            sns.boxplot(data=self.df, x=column, ax=axes[i, 0])
            axes[i, 0].set_title(f'Boxplot - {column}')
            sns.histplot(data=self.df, x=column, ax=axes[i, 1], bins=10)
            axes[i, 1].set_title(f'Histogram - {column}')
            axes[i, 0].tick_params(axis='x', labelrotation=45)
            axes[i, 1].tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.show()
        
    def numeric_discrete_to_target(self):
        if len(self.numeric_discrete_names) == 0:
            print("No data available for plotting.")
            return
        is_numeric_target = self.target.dtype in ['float64', 'int64']
        is_binary_target = len(np.unique(self.target)) == 2 if is_numeric_target else False
        is_continuous_target = is_numeric_target and not is_binary_target
        for col in self.numeric_discrete_features.columns:
            plt.figure(figsize=(10, 6))
            if is_continuous_target:
                sns.boxplot(x=col, y=self.target, data=self.df)
                plt.title(f'Boxplot of {col} vs Continuous Target')
                plt.show()
                groups = [self.target[self.df[col] == val] for val in self.df[col].unique()]
                f_stat, p_val = f_oneway(*groups)

                print(f"ANOVA F-Statistic for {col}: {f_stat}")
                print(f"ANOVA p-value for {col}: {p_val}")
                print("ANOVA tests the null hypothesis that the means of multiple groups are equal.")
                print("A low p-value (< 0.05) indicates that at least one group mean is different from the others.")
                print("A high p-value (> 0.05) suggests that the means are not significantly different.\n")
            elif is_binary_target:
                sns.countplot(x=col, hue=self.target, data=self.df)
                plt.title(f'Stacked Bar Chart of {col} vs Boolean Target')
                plt.show()
                contingency_table = pd.crosstab(self.df[col], self.target)
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                print(f"Chi-Square Test p-value for {col}: {p_val}")
                print("The Chi-Square Test tests the null hypothesis that the variables are independent.")
                print("A low p-value (< 0.05) indicates that the variables are not independent.")
                print("A high p-value (> 0.05) suggests that the variables are independent.\n")

    """All Things Categorical"""              
    
    def categorical_binary_describe(self):
        return self.categorical_binary_features.describe().T
    
    def categorical_binary_nulls(self):
        # Check if there are any binary categorical columns
        if not self.categorical_binary_names:
            return "No binary categorical columns found. Cannot plot null values."
        # Calculate null percentages for binary categorical columns
        null_percentages = round(self.df[self.categorical_binary_names].isnull().sum() / len(self.df) * 100, 3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns=column)
        return temp_df
    
    def categorical_binary_plot(self):
        if not self.categorical_binary_names:
            print("No binary categorical columns found. Cannot perform visualization.")
            return

        # Create subplots dynamically based on the number of binary features
        nrows = len(self.categorical_binary_names)
        ncols = 2  # Two plots for each binary feature
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 7 * nrows))

        # If only one row of subplots, axes is not a 2D array, so we reshape it
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(self.categorical_binary_names):
            # Vertical Bar Plot
            sns.countplot(x=col, data=self.df, ax=axes[i, 0])
            axes[i, 0].set_title(f'Frequency Distribution of {col}')

            # Point Plot
            sns.pointplot(x=col, y=self.df.index, data=self.df, orient='v', ax=axes[i, 1])
            axes[i, 1].set_title(f'Point Plot of {col}')

            # Annotate the bars with counts for the bar plot
            for p in axes[i, 0].patches:
                height = p.get_height()
                if pd.notna(height):
                    axes[i, 0].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()
       
    def categorical_binary_to_target(self):
        is_numeric_target = self.target.dtype in ['float64', 'int64']
        is_binary_target = len(np.unique(self.target)) == 2 if is_numeric_target else False
        is_continuous_target = is_numeric_target and not is_binary_target

        for col in self.categorical_binary_features.columns:
            if is_continuous_target:
                fig, axes = plt.subplots(1, 2, figsize=(16, 4))  # Create subplots
                
                # Boxplot
                sns.boxplot(x=col, y=self.target.apply(lambda x: custom_log(x)), data=self.df, ax=axes[0])
                axes[0].set_title(f'Boxplot of {col} vs Continuous Target')
                
                # Distribution Chart
                sns.histplot(x=self.target.apply(lambda x: custom_log(x)), 
                                                 hue=self.df[col], 
                                                 element="step", 
                                                 stat="density", 
                                                 common_norm=False, ax=axes[1])
                axes[1].set_title(f'Distribution of Target by {col}')
                
                plt.tight_layout()
                plt.show()
                
                groups = [self.target[self.df[col] == val] for val in self.df[col].unique()]
                f_stat, p_val = f_oneway(*groups)
                print(f"ANOVA F-Statistic for {col}: {f_stat}")
                print(f"ANOVA p-value for {col}: {p_val}")
                print("ANOVA tests the null hypothesis that the means of multiple groups are equal.")
                print("A low p-value (< 0.05) indicates that at least one group mean is different from the others.")
                print("A high p-value (> 0.05) suggests that the means are not significantly different.\n")

            elif is_binary_target:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=col, hue=self.target, data=self.df)
                plt.title(f'Stacked Bar Chart of {col} vs Boolean Target')
                plt.show()
                
                contingency_table = pd.crosstab(self.df[col], self.target)
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                print(f"Chi-Square Test p-value for {col}: {p_val}")
                print("The Chi-Square Test tests the null hypothesis that the variables are independent.")
                print("A low p-value (< 0.05) indicates that the variables are not independent.")
                print("A high p-value (> 0.05) suggests that the variables are independent.\n")


    def categorical_nominal_describe(self):
        return self.categorical_nominal_features.describe().T
    
    def categorical_nominal_nulls(self):
        # Check if there are any binary categorical columns
        if not self.categorical_nominal_names:
            return "No binary categorical columns found. Cannot plot null values."
        # Calculate null percentages for binary categorical columns
        null_percentages = round(self.df[self.categorical_nominal_names].isnull().sum() / len(self.df) * 100, 3)
        column = ["Percent Null"]
        temp_df = pd.DataFrame(null_percentages, columns=column)
        return temp_df
    
    def categorical_nominal_plot(self, variables=None):
        if variables is None:
            variables_to_plot = self.categorical_nominal_names
        elif isinstance(variables, list):
            variables_to_plot = self.categorical_nominal_names.drop(variables)
        else:
            raise ValueError("The variables input must be a list or None.")

        num_features = len(variables_to_plot)

        # Check if there is any data to plot
        if num_features == 0:
            print("No data available for plotting.")
            return

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_features, figsize=(10, 7 * num_features))

        # If there's only one subplot, axes will not be an array; make it into a list for consistency
        if num_features == 1:
            axes = [axes]

        for i, column in enumerate(variables_to_plot):
            sns.countplot(data=self.df, x=column, ax=axes[i], order=self.df[column].value_counts().index, palette=my_colors)
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
    
    def categorical_nominal_to_target(self):
        # Check if the list of nominal features is empty
        if len(self.categorical_nominal_names) == 0:
            print("No nominal features available for plotting.")
            return

        is_numeric_target = self.target.dtype in ['float64', 'int64']
        is_binary_target = len(np.unique(self.target)) == 2 if is_numeric_target else False
        is_continuous_target = is_numeric_target and not is_binary_target

        for col in self.categorical_nominal_names:
            if is_binary_target:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=col, hue=self.target, data=self.df)
                plt.title(f'Stacked Bar Chart of {col} vs Binary Target')
                plt.xticks(rotation=90)  # Rotate labels by 90 degrees
                plt.show()

                # Chi-Square Test
                contingency_table = pd.crosstab(self.df[col], self.target)
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                print(f"Chi-Square Test p-value for {col}: {p_val}")
                print("The Chi-Square Test assesses the independence between the nominal feature and the binary target.")
                print("A low p-value (< 0.05) suggests that the variables are associated or dependent.")
                print("A high p-value (> 0.05) suggests that the variables are independent.\n")

            elif is_continuous_target:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=col, y=self.target, data=self.df)
                plt.title(f'Boxplot of {col} vs Continuous Target')
                plt.xticks(rotation=90)  # Rotate labels by 90 degrees
                plt.show()

                # ANOVA Test
                groups = [self.target[self.df[col] == val] for val in self.df[col].unique()]
                f_stat, p_val = f_oneway(*groups)
                print(f"ANOVA F-Statistic for {col}: {f_stat}")
                print(f"ANOVA p-value for {col}: {p_val}")
                print("The ANOVA test evaluates whether the means of the target variable are significantly different across the groups in the nominal feature.")
                print("A low p-value (< 0.05) indicates that at least one group mean is different from the others.")
                print("A high p-value (> 0.05) suggests that there is no significant difference between the group means.\n")

                # Tukey HSD Post-hoc test
                if p_val < 0.05:
                    tukey = pairwise_tukeyhsd(endog=self.target, groups=self.df[col], alpha=0.05)
                    print("Tukey HSD Post-hoc Test Results:")
                    print(tukey)
                    print("The Tukey HSD test compares the means of all possible pairs of groups.")
                    print("It adjusts the p-value to account for multiple comparisons, thereby controlling the Type I error rate.")
                    print("If the 'reject' column is True, it means the means of those two groups are significantly different.\n")
                
    def categorical_narrative_clustering(self, method='kmeans', n_clusters=3, eps=.8, min_samples=30):
        if self.categorical_narrative_features.empty:
            print("The narrative group is empty. Cannot perform clustering.")
            return

        for col in self.categorical_narrative_features.columns:
            if self.df[col].isna().any():
                print(f"Column {col} contains NaN values. Skipping this column.")
                continue

            try:
                vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english")
                X = vectorizer.fit_transform(self.df[col])

                if method == 'kmeans':
                    model = KMeans(n_clusters=n_clusters)
                elif method == 'dbscan':
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                else:
                    print("Invalid method. Choose either 'kmeans' or 'dbscan'.")
                    return

                model.fit(X)
                cluster_labels = model.labels_
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X.toarray())

                plt.figure(figsize=(10, 6))

                unique_labels = set(cluster_labels)
                num_clusters = len(unique_labels)

                for i in unique_labels:
                    plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1], 
                                label=f'Cluster {i}' if num_clusters <= 10 else '')

                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.title(f'{method.upper()} Clustering of {col}')

                if num_clusters <= 10:
                    plt.legend()  # Show distinct legend
                else:
                    plt.colorbar(plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap= my_cmap))  # Show continuous color bar

                plt.show()

                print("How to Read the Legend:")
                print("1. Each label in the legend corresponds to a cluster.")
                print("2. Points in the scatter plot with the same color belong to the same cluster.")
                print("3. The label 'Cluster 0', 'Cluster 1', etc., indicates the cluster number.")
                print("4. If the number of clusters is above 10, a continuous color bar is used instead of distinct labels.")

            except ValueError as e:
                print(f"An error occurred while processing column {col}: {e}")
                continue
        # Create the plot
        sns.displot(kind='kde',
                    data=df_,
                    col='Distribution',
                    col_wrap=3,
                    x='value',
                    hue='Target',
                    fill=True,
                    height=8,
                    aspect=1.5,
                    facet_kws={'sharey': False, 'sharex': False}
                    )

        sns.pairplot(df_, hue = 'Target', diag_kind = 'kde', markers=["o", "s"])

