import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Hours":[1, 2, 3, 4, 5, 6, 7, 8],
    "Marks":[35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

model = LinearRegression()
model.fit(df[["Hours"]],df["Marks"])

hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])

print("Predicted Marks:",predicted_marks[0])

plt.scatter(df["Hours"],df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()
