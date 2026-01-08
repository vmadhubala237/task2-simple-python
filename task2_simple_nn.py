import random

# Sample dataset (Age, Attendance) -> Marks
data = [
    ([20, 85], 78),
    ([21, 90], 88),
    ([19, 70], 65),
    ([22, 95], 92),
    ([20, 80], 75)
]

# Initialize weights and bias
w1 = random.random()
w2 = random.random()
bias = random.random()
learning_rate = 0.001

# Activation function
def relu(x):
    return max(0, x)

# Training loop
for epoch in range(100):
    total_error = 0

    for inputs, target in data:
        x1, x2 = inputs

        # Forward pass
        prediction = relu(w1 * x1 + w2 * x2 + bias)

        # Error
        error = target - prediction
        total_error += abs(error)

        # Backpropagation (manual)
        w1 += learning_rate * error * x1
        w2 += learning_rate * error * x2
        bias += learning_rate * error

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Error: {total_error}")

# Test prediction
test_age = 21
test_attendance = 88
result = relu(w1 * test_age + w2 * test_attendance + bias)

print("\nPredicted Marks:", round(result))
