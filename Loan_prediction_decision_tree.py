# LOAN-APPROVAL kaggle dataset
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# step1 : load kiya dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")
print(df)
"""step2 : drop loan_id column becoz ml ke liye meaningful nai hai, agar drop nai kiya toh model uss column ko 
bhi training mein use karega jo galat learning dega"""
df.drop('Loan_ID', axis=1, inplace=True)
print(df.head())
#step3 : drop rows of missing values
df = df.dropna()

#step4 : manually mapping of categorical columns
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
df['Married'] = df['Married'].map({'Yes':1,'No':0})
df['Property_Area'] = df['Property_Area'].map({'Urban':2,'Semiurban':1,'Rural':0})
df['Self_Employed'] = df['Self_Employed'].map({"Yes":1,"No":0})
#step5 : split in features and targets 
x = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area']] # input mein sab ayega loan_status chhod ke 
y = df['Loan_Status'] # loan_status apna output hai

#step6 : train_test split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#step7 : train the decision tree model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

#step8 : make prediction
y_pred = model.predict(x_test)

#step9 : evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy: {accuracy * 100:.2f}%")

#step10: visualise decision tree 
plt.figure(figsize=(15,8)) # adjust the size of the plot
plot_tree(
    model,
    filled=True,
    feature_names=x.columns,
    class_names=["No","Yes"]
)
plt.title("decision tree for loan approval               ",fontsize=16)
plt.show()
