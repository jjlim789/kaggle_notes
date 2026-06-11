# Intermediate Machine Learning

## Notes

### Missing Values
- Just drop the column
- **Imputation**: Fill the missing value with a sentinel value.
- You can also add another column which is a boolean "did this row have a missing value". Depending on the column, different imputations (zero/mean/median/mode) makes more sense.
- Be careful to choose your imputation value ONLY on the training set and apply this to the test set. For example, when doing the imputation on the test set, don't take the mean of the values of the test set (this is looking forward)

### Categorical Variables
- Just drop the column
- **Numerical encoding**: Makes sense when the categories have a sense of "nearness" to each other.
- **One hot encoding**: Make a boolean column for every possible categorical variable (only works for low-nunique columns)

### Gradient boosting and XGBoost
- Start with a single naive model. Generate some predictions, calculate loss function. Train a new model that would reduce this loss (i.e. gradient descent). Add new model to the ensemble, and repeat.
- `XGBRegressor` parameters:
    - `n_estimators`: How many models should we add to the ensemble?
    - `learning_rate`: We multiply each prediction from each model by this number (learning_rate < 1.0), which means we can set a higher value for `n_estimators` without overfitting. Usually this results in more accurate XGB models, but longer training times. Default is `0.1`.
- `XGBRegressor.fit` parameters:
    - `early_stopping_rounds`: If used, we use the validation data to figure out when to stop adding to the ensemble (capped at `n_estimators`). The parameter itself determines how many rounds of validation score deteriation we need to see before we stop iterating. If used, we need to provide `eval_set=[(X_valid, y_valid)]`. Watch out for leakage; we are using the validation set to affect training. Maybe we need to cut out another test set?

### Target leakage
- Do not use features that are obtained after the target value is realised (i.e. do not look into the future)
- Do not use features that have the target information leaked into them; i.e. looking into the future again.

## Patterns Learned

### Imputation
```python
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

### Ordinal Encoding
```python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)
```

### Pipelines
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)
                              ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

### Cross validation
```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(
    my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error'
)
```
