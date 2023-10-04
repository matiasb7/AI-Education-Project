import pandas as pd
import numpy as np
from MachineLearning import MachineLearning
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def pre_clean(df):
    df['Subscription'].replace(
        {1: 0,
         2: 1}, inplace=True)

    yes_no_cols = ['Default', 'Loan', 'Housing Loan']
    for col in yes_no_cols:
        df[col].replace(
            {'yes': 1,
             'no': 0}, inplace=True)

    return df


def item1(df):
    if isinstance(df, pd.DataFrame):
        return {
            'Cantidad de (filas,columnas)': df.shape,
            'Existen valores vacios?': df.isnull().values.any(),
            'Cantidad de valores vacios': df.isna().values.sum(),
            'Columnas categoricas': ['Default', 'Loan', 'Housing Loan', 'Subscription', 'Marital Status', 'Education',
                                     'Job', 'Contact', 'Poutcome'],
            'Gente casada/soltera/divorciada': df['Marital Status'].value_counts(),
            'Educacion de personas': df['Education'].value_counts(),
            'Subscripción al plazo fijo': df['Subscription'].value_counts()
        }
    return ''


def split_cols(df, columns_to_dummies):
    for column in columns_to_dummies:
        df_dummies = df[column].str.get_dummies().add_prefix(column + '_')
        df = pd.concat([df, df_dummies], axis=1)

    df.drop(columns_to_dummies, axis=1, inplace=True)
    return df


def item2(df):
    # Work missing values:


    # Replace Age with normal distribution
    index = df[df['Age'].isna()].index
    value = np.random.normal(loc=df['Age'].mean(), scale=df['Age'].std(), size=df['Age'].isna().sum())
    value[value < 18] = df['Age'].mean()  # Replace values under 18
    df['Age'].fillna(pd.Series(value, index=index), inplace=True)

    # Drop the rest of empty values
    df = df.dropna()

    # Split categorical columns
    columns_to_dummies = ['Marital Status', 'Education', 'Job', 'Contact', 'Poutcome']
    df = split_cols(df, columns_to_dummies)

    columns_to_drop = ['Last Contact Month', 'Last Contact Day']
    df.drop(columns_to_drop, axis=1, inplace=True)

    # See correlation to select the variables for the model (if we use linear regression)
    corr_cols = []
    corr_dic = {}
    min_correlation = 0.1

    for column in df.columns:
        corr_value = df['Subscription'].corr(df[column])
        if corr_value > min_correlation or corr_value < -min_correlation:
            corr_cols.append(column)
            corr_dic[column] = corr_value

    # Graphics to select variables:

    # Selected: Last Contact Duration, Poutcome_success
    # Get Subscription correlations map
    # sns.heatmap(df[corr_cols].corr(), annot=True, cmap="YlGnBu")
    # plt.show()

    # sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    # plt.show()

    # from sklearn.feature_selection import mutual_info_classif
    # importances = mutual_info_classif(df.loc[:,df.columns != 'Subscription'],df['Subscription'])
    # feat_importances = pd.Series(importances, df.columns[0:len(df.columns)-1])
    # feat_importances.plot(kind="barh", color="teal")
    # plt.show()


    return df


def item4(train_df, test_df, ml_cols):
    # Set up data
    train_df_x = train_df[ml_cols]
    train_df_y = train_df['Subscription']

    # Splt categorical column
    test_df_clean = split_cols(test_df, ['Poutcome'])
    test_df_x = test_df_clean[ml_cols]

    ml = MachineLearning(train_df_x, train_df_y)

    # Comparison of the 3 chosen models
    model_tests = ml.results
    print(model_tests)

    # Predict Y with the test CSV using the winning model: random forest.
    output_y = ml.predict_data('random-forest', test_df_x)
    test_df['y-Subscription'] = output_y

    return test_df


# ----------------------------------------------------------------------------------------------------------------------

# Read Files
train_df = pd.read_csv('input/Bank Marketing train_set.csv')
test_df = pd.read_csv('input/Bank Marketing test_set.csv')

# Replace Yes/No for (0,1)
df = pre_clean(train_df)

# Find all items for answering item 1.
item1_answers = item1(df)

# Select columns to work with
df = item2(df)

# After item2, columns selected for the model are: 'Last Contact Duration' and 'Poutcome_success'. Poutcome_success
# is the sub-variable from Poutcome where the result is 1 for success and 0 for no success.
final_results = item4(df, test_df, ['Last Contact Duration', 'Poutcome_success', 'Pdays'])  # Random forest wins with 90k ~

# Export file
final_results.to_csv('output/Bank Marketing test_set.csv', index=False)
