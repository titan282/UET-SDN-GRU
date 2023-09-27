import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout
import pickle
import tensorflow as tf
import pylab as pl
number_of_samples = 50000

data_attack = pd.read_csv('dataset_attack_training_data.csv', nrows=number_of_samples)
data_normal = pd.read_csv('dataset_normal_training_data.csv', nrows=number_of_samples)

data_normal.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len',
                        'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
                        'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
                        'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
                        'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                        'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
                        'tcp.time_delta', 'class']
data_attack.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len',
                        'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
                        'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
                        'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
                        'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                        'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
                        'tcp.time_delta', 'class']

# data_normal = data_normal.drop([ 'ip.hdr_len',
#                         'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
#                         'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
#                         'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
#                         'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
#                         'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
#                         'tcp.time_delta'], axis=1)
# data_attack = data_attack.drop(['frame.protocols', 'ip.hdr_len',
#                         'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
#                         'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
#                         'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
#                         'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
#                         'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
#                         'tcp.time_delta'], axis=1)

# features = ['frame.len', 'ip.src', 'ip.dst', 'frame.protocols']

features = ['frame.len']

X_normal = data_normal[features].values
X_attack = data_attack[features].values
Y_normal = data_normal['class']
Y_attack = data_attack['class']
X = np.concatenate((X_normal, X_attack))
Y = np.concatenate((Y_normal, Y_attack))

scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(X)
X = scalar.transform(X)
pickle.dump(scalar, open('fiscaler.pkl', 'wb'))
print("fiscaler.pkl created!")


for i in range(0, len(Y)):
    if Y[i] == "attack":
        Y[i] = 0
    else:
        Y[i] = 1

features = len(X[0])
samples = X.shape[0]
train_len = 25
input_len = samples - train_len
I = np.zeros((samples - train_len, train_len, features))

for i in range(input_len):
    temp = np.zeros((train_len, features))
    for j in range(i, i + train_len - 1):
        temp[j - i] = X[j]
    I[i] = temp

# print(A[0])

# Fitting SVM with the training set
# create and fit the GRU network
X_train, X_test, Y_train, Y_test = train_test_split(I, Y[25:100000], test_size=0.2, random_state=4)

model = Sequential()
model.add(Bidirectional(GRU(64, activation='tanh', kernel_regularizer='l2')))

model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid', kernel_regularizer='l2'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
X_train= tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train=tf.convert_to_tensor(Y_train, dtype=tf.float32)
history = model.fit(X_train, Y_train, epochs=1, validation_split=0.2, verbose=1)

model.save('brnn_model.h5')
# model.save('brnn_model.keras')

# classifier = SVC(kernel='linear', random_state=0)
# classifier.fit(X_train, y_train)

# Testing the model by classifying the test set
Y_pred = model.predict(X_test)
Y_pred = np.where(Y_pred.flatten() > 0.5, 1, 0)
Y_test = [int(label) for label in Y_test]

# # Creating confusion matrix for evaluation
cm = confusion_matrix(Y_test, Y_pred)
cr = classification_report(Y_test, Y_pred)

# # Print out confusion matrix and report
# print(y_pred)
print(cm)
print(cr)
print(accuracy_score(Y_test, Y_pred))

# # Export model
# # filename = 'classifier.sav'
# # joblib.dump(classifier, filename)
# # print("Model exported!")
