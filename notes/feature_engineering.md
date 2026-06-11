# Feature engineering

## Notes
### When feature engineering is needed
- Example: you want to predict the tensile strength of cement. Instead of using `water` and `cement` quantities as a feature, its ratio `water/cement` might be more indicative.

### Mutual Information
- We want to check how "correlated" a feature is with a target. Pearson correlation can only measure linear correlation, but mutual information can measure any correlation.
- Easy to interpret, computationally cheap, resistant to overfitting, theoretically sound(?). However it cannot detect interactions between features; it is a univariate metric.
- A feature is only useful to the extent that its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. You may need to transform the feature first to expose the association.

### Creating features
- Mathematical transforms involving one or more features
- Building up or breaking down features (string manipulation)
- Counting boolean columns horizontally
- Group aggregations
- Tips
    - Linear models can learn sums and differences easily
    - Linear and neural nets are better with normalised features
    - Most models aren't very good at ratios
    - Tree models are good at learning most relations but it's good to have it explicitly created, especially if data is limited.
    - Counts are especially helpful for tree models since it's not good at aggregating across many features at once.

### Clustering
- An unsupervised algorithm; there is no target involved, its goal is to find a structure of the features. It is a feature discovery technique.
- Cluster feature is categorical but outputs an integer label encoding. Might be good to use one-hot encoding in some cases
- K-means clustering is an algorithm where it chooses K centroids and clusters the points into K voronoi cells. Initilises the centroids randomly, move centroid to minimise total (Euclidean) distance. It is sensitive to scale.
    - `n_clusters`: Number of centroids to choose
    - `max_iter`: The max number of times to move the centroid to find optimality
    - `n_init`: Number of times to rerun the algo since a bad choice of initial centroids could affect optimality
    - `fit_predict` gets you the group label. `fit_transform` gets you the distance from the group

### Principal component analysis
- Helps discover features. If a component has the form `component = feature_1 - feature_2`, it might be good to try `feature_1/feature_2` as a feature.
- Dimensionality reduction: Low or near-zero variance components are often redundant
- Anomaly detection: Unusual variation might appear on low-variance components
- Noise reduction: example is a collection of sensor readings. It can collect the signals while isolating out the variance
- Decorrelation for ML models that struggle with correlated features.
- PCA only works with numeric data, sensitive to scaling and outliers.

### Target encoding
- For categorical features with large cardinality, it's good to numerically encode them by their relation to the target. Naively this is just a groupby.mean(target)
- However this might cause overfitting so we want to do `m-smoothing`, where we apply the following factor
    - `encoding = weight * category_mean + (1 - weight) * total_mean`
    - `weight = n/(n+m)` where `n` is the number of samples for this category, `m` is a smoothing parameter.
- This is a strong source of data leakage. NEED TO CUT OUT A DIFFERENT SAMPLE OF DATA TO FIT THE TARGET ENCODING (i.e. pre-training dataset)

## Patterns Learned

### Mutual Information
```python
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
```

### Mutual Information interacting features
```python
feature = xaxis_feature
sns.lmplot(
    x=feature, y=target, hue=separation_feature, col=separation_feature,
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
```

### Interacting a categorical feature with a continuous feature
```python
X_2 = pd.get_dummies(X['BldgType'], prefix='Bldg')
X_2 = X_2.mul(X['GrLivArea'], axis=0)
```

### K-means clustering
```python
from sklearn.cluster import KMeans
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X_scaled)
X["Cluster"] = X["Cluster"].astype("category")

X_cd = kmeans.fit_transform(X_scaled)
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
```

### Principal component analysis
```python
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs
```

```python
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
```

```python
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings
```

### Target encoding
```python
from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)
encoder.fit(X_encode, y_encode)
X_train = encoder.transform(X_pretrain)
```