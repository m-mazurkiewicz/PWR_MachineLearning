import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/raw/train.csv')

df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
df['Embarked'] = df['Embarked'].fillna('S')
df['Fare'] = df['Fare'].fillna(lambda x: df[df['Pclass'] == x['Pclass']]['Fare'].median())
age_avg = df['Age'].mean()
age_std = df['Age'].std()
age_null_count = df['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
# Next line has been improved to avoid warning
df.loc[np.isnan(df['Age']), 'Age'] = age_null_random_list
df['Age'] = df['Age'].astype(int)
df.drop(["Name", "Ticket", "PassengerId", "Cabin"], axis=1, inplace=True)

train_test_ratio = 0.8
train_idx = np.random.choice(df.index, size=int(train_test_ratio*df.shape[0]), replace=False)

os.makedirs('data/processed/', exist_ok=True)
df[df.index.isin(train_idx)].to_csv('data/processed/train.csv', index=False)
df[~df.index.isin(train_idx)].to_csv('data/processed/test.csv', index=False)


