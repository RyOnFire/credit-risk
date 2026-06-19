import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv('cs-training.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop(df[df['age'] == 0].index)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    df = df.drop(df[df['DebtRatio'] > 10].index)

    return df


def prepare_data(df):
    labels = df['SeriousDlqin2yrs']
    features = df.drop('SeriousDlqin2yrs', axis=1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
