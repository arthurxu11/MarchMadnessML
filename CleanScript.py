import pandas as pd
import numpy
from sklearn.linear_model import LinearRegression

# Import data
data = pd.read_csv("data4.csv")

# Get test group
test = data.sample(frac =.2)

# Group teams by their result. 7 meaning the team won the championship, 6 meaning the team lost in the finals, ...
result = data.groupby('Result').mean()
print(result)

# Print the r^2 value between result and different factors
for x in result.columns:
    if x != "Team" or x!= "Year" or x!= "Result":
        correlation_matrix = numpy.corrcoef(result.index, result[x])
        correlation_xy = correlation_matrix[0,1]
        print(x+" "+str(correlation_xy ** 2))

# Seperating the data, only using the stats with a r^2 value of 0.7 or higher
array = data.values
x = array[:,3:19]
y = array[:,2]
y=y.astype('int')

# Making a linear regression model
model = LinearRegression().fit(x, y)

totalloss = 0
count = 0

# For each team in the test group, predict their result, and add to the total loss, if necessary
array = test.values
for x in array:
    try:
        info = [x[3:19]]
        predictions = round(model.predict(info)[0], 3)
        print(str(x[0])+" "+str(x[1])+" Prediciton: "+str(predictions)+" Reality: "+str(x[2]))
        totalloss += (x[2]-predictions)**2
        count += 1
    except:
        pass

# Print the total loss for the test group
print(totalloss/count)

# The loss seems to stay from 1 - 1.2, although is seems to underestimate teams. It rarely ever gives a prediction of 4 or above.
# However, if you order the teams by prediction score, and make a bracket based off the ranking, it gives a good prediction.
# If the model predicted the 2019 March Madness bracket, it would have scored 1290 in the ESPN bracket, which would have placed it in the 98th percentile of all brackets.