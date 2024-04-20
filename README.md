import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("dataset.csv")

# Split features and target variable
X = data.drop(columns=["target_column"])
y = data["target_column"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply preprocessing to test data
X_test_preprocessed = preprocessor.transform(X_test)

# Optionally perform dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_preprocessed = pca.fit_transform(X_train_preprocessed)
X_test_preprocessed = pca.transform(X_test_preprocessed)

# Now, X_train_preprocessed and X_test_preprocessed are ready for machine learning algorithms
