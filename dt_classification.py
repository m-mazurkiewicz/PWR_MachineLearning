from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import graphviz

df_train = pd.read_csv('data/processed/train.csv')
df_test = pd.read_csv('data/processed/test.csv')

df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

max_depth = 4
random_state = 100

dt1 = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=2, min_impurity_decrease=0.0, presort=True, random_state=random_state)

dt1 = dt1.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
accuracy = dt1.score(df_test.drop(['Survived'], axis=1), df_test['Survived'])
print(accuracy)


dot_data = export_graphviz(dt1, out_file=None,feature_names=df_train.drop(['Survived'], axis=1).columns,
                        class_names=['0','1'],
                        filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("dt1")

dt2 = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=2, min_impurity_decrease=0.0, presort=True, random_state=random_state)

dt2 = dt2.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
accuracy = dt2.score(df_test.drop(['Survived'], axis=1), df_test['Survived'])
print(accuracy)

dot_data = export_graphviz(dt2, out_file=None,feature_names=df_train.drop(['Survived'], axis=1).columns,
                        class_names=['0','1'],
                        filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("dt2")
