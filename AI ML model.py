import csv
import os
import math


# This function creates a dataset file if it does not already exist
def create_dataset():
    folder = "../data"
    file_path = os.path.join(folder, "student_data.csv")

    # Create folder if not present
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Sample dataset
    # Last column: 1 = procrastinates, 0 = does not
    data = [
        ["study", "sleep", "screen", "attendance", "assignments", "stress", "marks", "procrastination"],
        [6, 7, 3, 80, 5, 4, 75, 0],
        [2, 5, 8, 60, 2, 8, 45, 1],
        [7, 8, 2, 90, 6, 3, 85, 0],
        [3, 6, 7, 65, 3, 7, 50, 1],
        [5, 7, 4, 75, 4, 5, 70, 0],
        [1, 4, 9, 50, 1, 9, 35, 1],
        [8, 8, 2, 95, 6, 2, 90, 0],
        [4, 6, 6, 70, 3, 6, 60, 1],
        [6, 7, 3, 85, 5, 4, 78, 0]
    ]

    # Write data into CSV file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Dataset ready\n")


# This function normalizes the dataset (values between 0 and 1)
def normalize(X):
    min_vals = []
    max_vals = []

    for j in range(len(X[0])):
        column = [row[j] for row in X]
        min_vals.append(min(column))
        max_vals.append(max(column))

    for i in range(len(X)):
        for j in range(len(X[0])):
            if max_vals[j] != min_vals[j]:
                X[i][j] = (X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
            else:
                X[i][j] = 0

    return X, min_vals, max_vals


# Normalize user input using same values
def normalize_input(user, min_vals, max_vals):
    new_user = []

    for i in range(len(user)):
        if max_vals[i] != min_vals[i]:
            value = (user[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
        else:
            value = 0
        new_user.append(value)

    return new_user


# Linear prediction (used for marks)
def predict(features, weights):
    result = weights[0]

    for i in range(len(features)):
        result += weights[i + 1] * features[i]

    return result


# Sigmoid function for probability
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Predict probability (used for procrastination)
def predict_procrastination(features, weights):
    value = predict(features, weights)
    return sigmoid(value)


# Training function using gradient descent
def train(X, y, learning_rate=0.01, iterations=5000):
    weights = [0.0] * (len(X[0]) + 1)

    for _ in range(iterations):
        for i in range(len(X)):
            prediction = predict(X[i], weights)
            error = y[i] - prediction

            # update bias
            weights[0] += learning_rate * error

            # update other weights
            for j in range(len(X[i])):
                weights[j + 1] += learning_rate * error * X[i][j]

    return weights


# Function to give suggestions based on result
def give_suggestion(marks, risk):
    if marks < 50 and risk >= 0.5:
        return "Low marks and high procrastination. Try reducing screen time and start studying in small steps."
    elif marks < 50:
        return "Marks are low. Increase study time and be more consistent."
    elif risk >= 0.5:
        return "You may procrastinate. Try better time management."
    else:
        return "Good performance. Keep following your routine."


# Function to assign grade
def get_grade(marks):
    if marks >= 85:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 50:
        return "C"
    else:
        return "D"


# Main program starts here
if __name__ == "__main__":
    create_dataset()

    X = []
    y_marks = []
    y_procrastination = []

    # Read dataset
    with open("../data/student_data.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            X.append([float(val) for val in row[:-2]])
            y_marks.append(float(row[-2]))
            y_procrastination.append(float(row[-1]))

    # Normalize data
    X, min_vals, max_vals = normalize(X)

    print("Training models...")
    weights_marks = train(X, y_marks)
    weights_pro = train(X, y_procrastination)
    print("Training completed\n")

    print("Enter your details:\n")

    user = [
        float(input("Study hours: ")),
        float(input("Sleep hours: ")),
        float(input("Screen time: ")),
        float(input("Attendance (%): ")),
        float(input("Assignments completed: ")),
        float(input("Stress level (1-10): "))
    ]

    # Normalize user input
    user = normalize_input(user, min_vals, max_vals)

    # Predictions
    marks = predict(user, weights_marks)
    marks = max(0, min(100, marks))  # keep within range

    risk = predict_procrastination(user, weights_pro)

    print(f"\nPredicted Marks: {marks:.2f}")
    print(f"Grade: {get_grade(marks)}")
    print(f"Procrastination Risk: {risk:.2%}")

    print(give_suggestion(marks, risk))