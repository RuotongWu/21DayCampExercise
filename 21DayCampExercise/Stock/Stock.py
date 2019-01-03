import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def read_data():
    """
    Read data from csv file.
    Using absolutate path, works on Windows

    Return: a dataframe objects contains the data read from the file
    """
    path = os.getcwd() + "\\Data\\msft_stockprices_dataset.csv"
    data = pd.read_csv(path)
    data = data.iloc[:,1:]                                      #drop the date column
    return data

def data_spliting(data):
    """
    Split data into training, validation and testing sets.
    Also split it into features and tag

    Returns: training features, training tag, validation features, validation tag, testing features, testing tag,
    """
    length = data.shape[0]

    training = data[:int(0.8*length)].reset_index(drop=True)
    training_tags = training[['Close Price']].values
    training_data = training.drop('Close Price',axis = 1).values

    validation = data[int(0.8*length):int(0.9*length)].reset_index(drop=True)
    validation_tags = validation[['Close Price']].values
    validation_data = validation.drop('Close Price',axis = 1).values

    testing = data[int(0.9*length):].reset_index(drop=True)
    testing_tags = testing[['Close Price']].values
    testing_data = testing.drop('Close Price',axis = 1).values

    return training_data, training_tags, validation_data, validation_tags, testing_data, testing_tags

def accuracy(ground_true, prediction, tolerance):
    """
    Calculate the accuracy

    Return: The accuracy
    """
    count = 0
    correct = 0

    for true, pre in zip(ground_true, prediction):
        count += 1
        if abs(true-pre)/true <= tolerance:
            correct += 1

    return correct/count

if __name__ == "__main__":

    data = read_data()
    data = data.sample(frac = 1).reset_index(drop = True)       # Shuffle the dataframe
    training_data, training_tags, validation_data, validation_tags, testing_data, testing_tags = data_spliting(data)

    
    lr = LinearRegression().fit(training_data,training_tags)    # Fit Linear Regression model
    
    validation_predicted = lr.predict(validation_data)          # Predict on validation set
    validation_accuracy = accuracy(validation_tags,validation_predicted, 0.05)

    testing_predicted = lr.predict(testing_data)                #predict on testing set
    testing_accuracy = accuracy(testing_tags,testing_predicted, 0.05)


    #plot figures

    fig, axs = plt.subplots(2,1, constrained_layout = True)
    fig.suptitle('The Results on the Validation Set and the Testing Set')

    red_patch = mpatches.Patch(color='red', label='The true Close Price')
    green_patch = mpatches.Patch(color='green', label='The predicted Close Price')
    fig.legend(handles=[red_patch, green_patch])

    validation_index = range(len(validation_tags))
    axs[0].plot(validation_index, validation_tags, 'r-o')
    axs[0].plot(validation_index, validation_predicted, 'g-o')

    axs[0].set_title("The Result on Validation Set (Accuracy: {:.2%})".format(validation_accuracy))
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Close Price")

    testing_index = range(len(testing_tags))
    axs[1].plot(testing_index, testing_tags, 'r-o')
    axs[1].plot(testing_index, testing_predicted, 'g-o')

    axs[1].set_title("The Result on Testing Set  (Accuracy: {:.2%})".format(testing_accuracy))
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Close Price")

    plt.show()

