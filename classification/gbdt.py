import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading the data...")
df = pd.read_csv(r'data\hate_speech\LSTM_random_embedding\mean_no_emiji.csv', encoding='utf-8')

print("Separating the classes...")
df_class_0 = df[df['class'] == 0]
df = df[df['class'].isin([1, 2])]

print("Preparing the features and labels...")
y = df['class'].astype(int)
X = df.drop('class', axis=1).values.tolist()

print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Defining the parameters for Grid Search...")
params = {'n_estimators': range(1,101,10)}

print("Initialising the GBDT Classifier...")
gbdt = GradientBoostingClassifier()

print("Performing Grid Search with Cross Validation...")
grid = GridSearchCV(gbdt, params, cv=5, scoring='accuracy', return_train_score=True)
grid.fit(X_train, y_train)

print("Best Parameters: ", grid.best_params_)

print("Plotting the results...")
plt.figure(figsize=(10, 6))
estimator_range = params['n_estimators']
train_scores_mean = grid.cv_results_['mean_train_score']
train_scores_std = grid.cv_results_['std_train_score']
test_scores_mean = grid.cv_results_['mean_test_score']
test_scores_std = grid.cv_results_['std_test_score']

plt.title('Accuracy as a function of number of trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')

plt.plot(estimator_range, train_scores_mean, label='Mean Train Score', color='darkorange', lw=2)
plt.plot(estimator_range, test_scores_mean, label='Mean Test Score', color='navy', lw=2)
plt.fill_between(estimator_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
plt.fill_between(estimator_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="navy")
plt.legend(loc="best")
plt.grid()
plt.show()

print("Predicting classes for df_class_0...")
best_gbdt = grid.best_estimator_
X_class_0 = df_class_0.drop('class', axis=1).values.tolist()
class_0_pred = best_gbdt.predict(X_class_0)
df_class_0['predicted_class'] = class_0_pred

print("Saving df_class_0 with predicted classes to CSV...")
df_class_0.to_csv(" predicted_class_0.csv", index=False, encoding='utf-8')

print("Displaying the distribution of predictions...")
sns.countplot(x='predicted_class', data=df_class_0)
plt.show()
