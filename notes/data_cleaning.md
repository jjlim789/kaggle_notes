# Data cleaning

## Notes
(This lesson was very boring other than the Boxcox transformation)

- Methods to address missing data
  - Is the data actually missing or is it just not applicable?
  - Drop missing data rows or columns
  - Impute data with sentinel value, or bfill/ffill
- Scaling and normalisation
  - Scaling (changing the range of data): Required when using methods based on measures of how far apart the datapoints are, like SVM or KNN
  - Normalisation (changign the shape of data): Required when the method assumes that the data is normally distributed, like LDA or Gaussian Naive Bayes
- Boxcox transformation is a normalising technique, given by a parameter $\lambda >=0$. $x \mapsto (x^\lambda - 1)/\lambda$
  - Preserves monotonicity, and it's injective (so it's invertible)
  - Stabilieses variance; I.e. makes data homoskedastic. Many models assume homoskedasticity.
  - The choice of lambda is estimated using the MLE approach which maximises the "log-likelihood that the data came from a normal distribution"
  - Better to get a confidence interval and choose a "clean number" like 0, 0.5, 1 or -1 which makes this transform more interpretable
- Parsing dates
- Character encoding: Sometimes files are encoded with bullshit encodings. See patterns below
- Inconsistent Data entry: Uses fuzzy matching to match string entries that are the same but use different formats
  - Basic techniques are `.lower()`, `.strip()`

## Patterns Learned

### Boxcox transformation
```python
from scipy import stats
normalized_data = stats.boxcox(original_data)
```

### Character encoding
```python
with open("filename", 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(30000))
# result will tell you the correct encoding
df = pd.read_csv(filename, encoding=encoding)
```

### Fuzzy matching
```python
import fuzzywuzzy
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
```