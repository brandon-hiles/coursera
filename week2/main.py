import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from preprocessed_mnist import load_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print(X_train.shape, y_train.shape)
plt.imshow(X_train[0], cmap="Greys");

# Reshape the training and test examples.
image_size = 28*28
X_train = X_train.reshape(X_train.shape[0], image_size) 
X_test = X_test.reshape(X_test.shape[0], image_size)

# Define placeholders
input_X = tf.placeholder(tf.float32, shape=(None, image_size))
input_y = tf.placeholder(tf.float32, shape=(None, 10)) # 10 different digits
keep_prob = tf.placeholder(tf.float32)

# Define Variables
W1 = tf.get_variable("W1",  [784, 128], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1",  [128], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2",  [128, 128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2",  [128], initializer=tf.zeros_initializer())
W3 = tf.get_variable("W3", [128, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3",  [10], initializer=tf.zeros_initializer())

# Run Logistic Regression

Z1 = tf.matmul(input_X, W1)+ b1
A1 = tf.nn.relu(Z1)
A1 = tf.nn.dropout(A1, keep_prob)
Z2 = tf.matmul(A1, W2)+b2
A2 = tf.nn.relu(Z2)
A2 = tf.nn.dropout(A2, keep_prob)
Z3 = tf.matmul(A2, W3)+b3
A3 = tf.nn.sigmoid(Z3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=A3))
starter_learning_rate = 0.0007 # Guess
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.85, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

init = tf.global_variables_initializer()
batch_size = 500
with tf.Session() as sess:
    sess.run(init)
    train_costs = []
    test_costs = []
    for epoch in range(100):
        for i in range(0, 50000, batch_size):
            sess.run(optimizer, feed_dict={input_X:X_train[i:i+batch_size], input_y:y_train[i: i+batch_size], keep_prob : 0.65})
        train_costs.append(sess.run(cost, feed_dict={input_X:X_train, input_y:y_train, keep_prob : 1}))
        test_costs.append(sess.run(cost, feed_dict={input_X:X_test, input_y:y_test, keep_prob : 1}))
        if epoch%10 == 9:
            print("Cost after " + str(epoch+1)+ " epochs: "+ str(train_costs[-1])) # Should be decreasing
    iterations = list(range(100))
    plt.plot(iterations, train_costs, label='Train')
    plt.plot(iterations, test_costs, label='Test')
    plt.ylabel('train cost')
    plt.xlabel('iterations')
    plt.show()
    
    # Calculate the correct predictions
    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(input_y, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy = accuracy.eval({input_X: X_train, input_y: y_train, keep_prob : 1})
    test_accuracy = accuracy.eval({input_X: X_test, input_y: y_test, keep_prob : 1})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)