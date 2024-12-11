import csv
import math
import random
import pandas as pd
import numpy as np


def readfile(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # The first column is assumed to be the label, and the rest are features
    labels = df.iloc[:, 0].values  # Get the labels (first column)
    data = df.iloc[:, 1:].values  # Get the feature data (all columns except the first)

    # Map labels (e.g., 'X1', 'X2') to numeric values 0 and 1 for binary classification
    label_mapping = {"X1": 0, "X2": 1, "X3": 1, "X4": 0, "X5": 0, "X6": 1}
    labels = [label_mapping[label] for label in labels]

    # # Convert categorical labels (e.g., 'X1', 'X2') to numeric values
    # label_mapping = {label: idx for idx, label in enumerate(set(labels))}
    # labels = [label_mapping[label] for label in labels]

    return labels, data


#Activation function

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron(dataInput,labels, alpha = 0.001, epochs=100):
    total_error = 0
    bias = random.uniform(-0.1,0.1)
    weight = [random.uniform(-0.5, 0.5) for _ in range(len(dataInput[0]))] #To have individuals weight for each feature

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(dataInput)):

            input = dataInput[i]
            target = int(labels[i])

            weighted_sum = sum(input[j] * weight[j] for j in range(len(input))) + bias

            # Using sigmoid as my activation function
            output = step_function(weighted_sum)

            # The difference between actual and predicted output
            error = target - output
            total_error += error ** 2

            for j in range(len(weight)):
                weight[j] += alpha * error * input[j]

            # Updated bias
            bias += alpha * error
        print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error}")


    return weight, bias

def predict(input_data, weights, bias):
    # Calculate the weighted sum of inputs
    weighted_sum = sum(input_data[i] * weights[i] for i in range(len(input_data))) + bias
    # Apply step function for binary classification
    return step_function(weighted_sum)



if __name__=="__main__":
    filename = "input.csv"
    labels, data = readfile(filename)
    print("Labels:", labels)
    print("Data:", data)

    # Train the perceptron
    weights, bias = perceptron(data, labels)

    print("Learned weights:", weights)
    print("Learned bias:", bias)

    print("New data")

    # Test on new data
    new_data = [[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]]  # Example new input
    prediction = predict(new_data[0], weights, bias)
    print("Prediction for new data:", prediction)

