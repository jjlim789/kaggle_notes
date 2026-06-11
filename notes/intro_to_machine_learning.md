# Intro to Machine Learning

## Notes

- **`max_leaf_nodes`**: Parameter in the regressor constructor class
  - Find a good value by iterating to find what minimises oos error
- **Random Forest** is often better than a decision tree

## Patterns Learned

### Data Loading & Exploration
```python
pd.read_csv("file.csv")

series.describe()

series.head()
```

### Decision Tree Model
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
test_pred = model.predict(test_X)
```

### Model Evaluation
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test_y, test_pred)
```

### Train/Validation Split
```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
```

### Random Forest Model
```python
from sklearn.ensemble import RandomForestRegressor
```
