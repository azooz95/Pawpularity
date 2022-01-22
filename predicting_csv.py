from library import *

csv_file = "new_file.csv"
# path = "D:\\Data\\animal"
# csv_file = os.path.join(path,'train.csv')
csv_data = pd.read_csv(csv_file, index_col=False)

train_calumn = csv_data

print(csv_data.columns)
data = csv_data.loc[:,csv_data.columns != 'Pawpularity']
data = data.loc[:,data.columns != 'Id']
print(data.keys())
labels = csv_data["Pawpularity"]

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=52)

# pipe = Pipeline([('svr', RandomForestRegressor(n_estimators= 300, random_state=0))])

# pipe.fit(train_x,train_y)

# y_pred = pipe.predict(test_x)

# print(f"and this is pred_y{y_pred}")
# print(f"and this is test_y{test_y}")

# a = mean_squared_error(test_y, y_pred)

# print(a)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf = RandomizedSearchCV(estimator = RandomForestRegressor(),n_iter=20,param_distributions=random_grid,scoring='neg_mean_squared_error', verbose=2)
clf.fit(train_x,train_y)
print(clf.best_params_)

y_pred = clf.predict(test_x)

a = mean_squared_error(test_y, y_pred)

print(a)