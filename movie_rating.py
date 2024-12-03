import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")
print(df.head())
print("Columns : ",df.columns)
#preprocessing the dataset
print("Missing Values:\n",df.isnull().sum())
df = df.dropna()
print(df.shape)
df = pd.get_dummies(df, columns=["Genre", "Director"], drop_first=True)
for col in df.select_dtypes(include=['object']).columns:
    print(f" Column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
target = "Rating"
if target not in df.columns:
    print(f"Column '{target}' not found. Available columns are: {df.columns}")
else:
    x = df.drop(target, axis=1)
    y = df[target]
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    df.to_csv("processed_movies_dataset.csv", index= False)
    print("Processed Dataset is saved !")
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

