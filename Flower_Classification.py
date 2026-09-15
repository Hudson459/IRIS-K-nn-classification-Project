import pandas as pd
import numpy as np
import math as math
import time as time
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
"""
  Hudson Whitbeck's Project implementing Knn classification to classify flowers based on surrounding data points
"""

"""
The hardcoded values in this script are:

number_of_training_rows(line 27)
number_of_testing_rows(line 28)
k -- Denoting the range from 0 - k neighbors that you want to run calculations for (line 41)
repeat -- Denoting the amount of times you want to calculate the correct classification percentage under the current conditions to get an overall average (line 46)
"""

# Get the folder where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build a path to the data folder safely
data_path = os.path.join(current_dir, "IRIS.csv")

# Importing the training and test data from my local hard drive
train = pd.read_csv(data_path)
train.info()

number_training_rows = 120
number_testing_rows = 150 - number_training_rows - 2


#Turning the training data into a numpy array so we can use it
data_full = train.to_numpy()



x = list()
y = list()

#We will gather data by repeating our predictions for every value of k considered neighbors
for k in range(1, 120):

    #Creating a list to collect the different correct prediction percentages for each k value
    repeated_percentage = list()

    #To get a better percentage value, we will predict multiple times with each k value, then take the average correct classification percentage
    for repeat in range(0, 10):

        #The data is organized by label, so we will shuffle it by row before hand
        np.random.shuffle(data_full)

        #Since the data is not standardized, we can split it into our training and test data
        train_data = data_full[1:number_training_rows, 1:]
        train_data_classifications = data_full[1:number_training_rows, 0]
        test_data  = data_full[number_training_rows+2:, 1:]
        test_data_classifications = data_full[number_training_rows+2:, 0]

        #Standardizing the data by column so all rows have a mean of 0 and a standard deviation of 1
        #Calculate the mean
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data-mean)/std
        test_data = (test_data-mean)/std
        #These numbers will be used to generate a correct percentage for our classifications
        total_guesses = 0
        total_correct = 0

        #We will run the test seperately for every input
        for test_num in range(0, number_testing_rows):

            #capturing the intended outcome so that we can compare our value to it later on
            intended_outcome = test_data_classifications[test_num]

            #the position the data point to be classified in space
            sample = test_data[test_num]

            #Creation of the data matrix, done by using the euclidian distance from every point in the training data
            distance = np.sum((train_data - sample)**2, axis=1)


            #We combine the distance matrix with the outcomes from the training data, so that we may sort the distances without losting the index of the intended result.
            values_distance = np.column_stack(( distance, train_data_classifications))

            #We will sort the values_distance array from least to greatest 
            values_sorted = values_distance[(values_distance[:, 0].astype(float)).argsort()]

            #We will now choose the k closest points and seperate them from the rest
            values_kth_neighbors = values_sorted[:k, :]


            #We will move through the nearest neighbors and overwrite each distance value with a number corrisponding to their outcome
            for index in range(0, len(values_kth_neighbors)):
                if str(values_kth_neighbors[index][1]) == 'Iris-versicolor':
                    values_kth_neighbors[index][0] = 0
                elif str(values_kth_neighbors[index][1]) == "Iris-setosa":
                    values_kth_neighbors[index][0] = 1
                elif str(values_kth_neighbors[index][1]) == "Iris-virginica":
                    values_kth_neighbors[index][0] = 2
                else:
                    print("PROBLEM WITH LOOP ON LINE 52")
                    quit()

            #We want to predict based on the most frequent near outcome, so we will take the mode of the three outcome labels and use that as our prediction
            prediction = stats.mode(values_kth_neighbors[:,0].astype(int))[0]

            #Printing and comparing our predictiont to the intended outcome we stored at the very beginning (Line 43)
            if np.round(prediction) == 0:
                """print("Prediction is Iris-versicolor ---- Actual outcome is: ", intended_outcome )"""
                if(intended_outcome == "Iris-versicolor"):
                    total_correct += 1
                total_guesses += 1
            elif np.round(prediction) == 1:
                """print("Prediction is Iris-setosa ---- Actual outcome is: ", intended_outcome)"""
                if(intended_outcome == "Iris-setosa"):
                    total_correct += 1
                total_guesses+=1
            elif np.round(prediction) == 2:
                """print("Prediction is Iris-virginica ---- Actual outcome is: ", intended_outcome)"""   
                if(intended_outcome == "Iris-virginica"):
                    total_correct += 1
                total_guesses += 1
            else:
                print("SOMETHING IS WRONG")

        #Total correct classification for this given run
        percentage = total_correct/total_guesses

        #We add this to the list to average at the end of the runs for this k value
        repeated_percentage.append(percentage)

    #After all the runs for a given k value are completed, we average them for a final percentage to be considered
    total_average = np.average(repeated_percentage)


    print("The total average percentage of correct classifications for ", k, " nearest neighbors is ", np.round(total_average, 4)*100, "%")
    y.append(total_average)
    x.append(k)

#The plot created shows the total accuracy of the model based on the number of neighbors considered
plt.plot(x, y)
plt.xlabel("Number of nearest neighbors considered")
plt.ylabel("Percentage of correct classifications")

plt.show()


