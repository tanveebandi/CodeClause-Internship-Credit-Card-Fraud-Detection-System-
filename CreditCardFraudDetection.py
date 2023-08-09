import pandas as pd #important for reading csv file
from sklearn.model_selection import train_test_split # this allows you to split the data in training and test datas 
from sklearn.linear_model import LogisticRegression #important for checking the accuracy 
from sklearn.metrics import accuracy_score #Important for checking the performance of the model

# Replace 'your_file_path/fraudTrain.csv' with the actual file path to your CSV file
file_path = r'C:\Users\gajen\OneDrive\Desktop\creditcard.csv'

# Read the CSV file into a pandas DataFrame
credit_card_data = pd.read_csv(file_path)

# Now you can work with the 'credit_card_data' DataFrame
print(credit_card_data.head())  # Display the first few rows of the DataFrame
print(credit_card_data.tail())
credit_card_data.info()
credit_card_data.isnull().sum()
print(credit_card_data['Class'].value_counts())
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)
print(legit.Amount.describe())
print(fraud.Amount.describe())
credit_card_data.groupby('Class').mean()
legit_sample=legit.sample(n=492)
new_dataset=pd.concat([legit_sample,fraud],axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())
# Assuming new_dataset is your DataFrame

# Assuming new_dataset is your DataFrame
X = new_dataset.drop('Class', axis=1)


Y=new_dataset['Class']
print(X)

# Assuming X and Y are your feature and target variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Now you can use X_train, X_test, Y_train, and Y_test in your further processing
print(X.shape,X_train.shape, X_test.shape)
print(Y.shape,Y_train.shape, Y_test.shape)


# Assuming X_train and Y_train are your training data
model = LogisticRegression()  # Create a Logistic Regression model

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Fit the model to the training data
print(model.fit(X_train, Y_train))
# Now the model is trained and you can use it for predictions
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data:", training_data_accuracy)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on test data',test_data_accuracy)

                                  
                                  
