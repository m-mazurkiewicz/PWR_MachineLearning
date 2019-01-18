from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import graphviz
from matplotlib import pyplot as plt

df_train = pd.read_csv('data/processed/train.csv')
df_test = pd.read_csv('data/processed/test.csv')

df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# max_depth = 4
random_state = 100
search_range = range(2, 10)


best_accuracy = 0
best_max_depth_1 = 0
acc_hist_1 = []
for max_depth in search_range:
    dt1 = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=2, min_impurity_decrease=0.0, presort=True, random_state=random_state)

    dt1 = dt1.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
    accuracy = dt1.score(df_test.drop(['Survived'], axis=1), df_test['Survived'])
    acc_hist_1.append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_max_depth_1 = max_depth
print(best_max_depth_1,best_accuracy)

plt.figure(figsize=(10,6), dpi=300)
plt.plot(search_range, acc_hist_1, 'r.', markersize = 20)
plt.xlabel("Maximum depth")
plt.ylabel("Accuracy on test set")
plt.savefig('acc_dt1.pdf')
plt.close()



dt1 = DecisionTreeClassifier(criterion='gini', max_depth=best_max_depth_1, min_samples_split=2, min_impurity_decrease=0.0,
                             presort=True, random_state=random_state)

dt1 = dt1.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
dot_data = export_graphviz(dt1, out_file=None,feature_names=df_train.drop(['Survived'], axis=1).columns,
                        class_names=['0','1'],
                        filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("dt1")

best_accuracy = 0
best_max_depth_2 = 0
acc_hist_2 = []
for max_depth in search_range:
    dt2 = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=2, min_impurity_decrease=0.0, presort=True, random_state=random_state)

    dt2 = dt2.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
    accuracy = dt2.score(df_test.drop(['Survived'], axis=1), df_test['Survived'])
    acc_hist_2.append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_max_depth_2 = max_depth
print(best_max_depth_2,best_accuracy)

plt.figure(figsize=(10,6), dpi=300)
plt.plot(search_range, acc_hist_2, 'r.', markersize = 20)
plt.xlabel("Maximum depth")
plt.ylabel("Accuracy on test set")
plt.savefig('acc_dt2.pdf')
plt.close()

dt2 = DecisionTreeClassifier(criterion='entropy', max_depth=best_max_depth_2, min_samples_split=2, min_impurity_decrease=0.0, presort=True, random_state=random_state)

dt2 = dt2.fit(df_train.drop(['Survived'], axis=1), df_train['Survived'])
dot_data = export_graphviz(dt2, out_file=None,feature_names=df_train.drop(['Survived'], axis=1).columns,
                        class_names=['0','1'],
                        filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("dt2")
