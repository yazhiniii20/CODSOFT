import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(train_data.head())
# print(train_data.info())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data = train_data.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin', 'Name'], axis=1)
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
print(train_data.isnull().sum())
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
x_train = train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y_train = train_data['Survived']
x_test = test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]


model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
print("Model trained successfully")
print(train_data.head())
print(train_data.info())
train_accuracy = model.score(x_train,y_train)
# val_accuracy = model.score(x_val,y_val)
print(f"Training accuracy: {train_accuracy}")
# print(f"Validation accuracy: {val_accuracy}")
