import pandas as pd

def load_data():
    df = pd.read_csv('cs-training.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop(df[df['age'] == 0].index)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    df = df.drop(df[df['DebtRatio'] > 10].index)
    return df
