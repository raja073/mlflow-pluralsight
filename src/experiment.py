import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load complete dataset, select features + target variables
data = pd.read_csv("../data/ames_housing.csv")
feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
selected = data.loc[:, feature_columns + ["SalePrice"]]

# Feature that need encoding (categorical features)
cat_features = ["Bldg Type"]


def prepare_data(dataframe):
    df = dataframe
    # Encode all the categorical features
    for col in list(df.columns):
        if col in cat_features:
            # One-hot encoding
            dummies = pd.get_dummies(df[col])
            # Drop the original column
            df = pd.concat([df.drop([col], axis=1), dummies], axis=1)
    # fill missing values with 0
    df = df.fillna(0)

    return df

def train_and_evaluate(dataframe):
    # Separate features from target variables and convert to NumPy
    features = dataframe.drop(['SalePrice'], axis=1).to_numpy() 
    target = dataframe.loc[:, "SalePrice"].to_numpy()
    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        deatures, target, test_size=0.2, random_state=12
    )

    # Train the model
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    print(f"MSE: {err}")

# Add None so that we get one training with all the columns
columns_to_drop = feature_columns + [None]

for to_drop in columns_to_drop:
    if to_drop:
        dropped = selected.drop([to_drop], axis=1)
    else:
        dropped = selected

    print(f"Dropping {to_drop}")
    prepared = prepare_data(dropped)
    train_and_evaluate(prepared)


