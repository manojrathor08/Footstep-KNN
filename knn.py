# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from data import DataSet
from sklearn.neighbors import KNeighborsClassifier
batch_size = 32
hidden_size = 10
use_dropout=True
vocabulary = 6
data_type = 'features'
seq_length = 60
class_limit =  6
image_shape = None
data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        ) 
# loading the iris dataset
X_tr, y_tr = data.get_all_sequences_in_memory('train', data_type)
X_train= X_tr.reshape(780,1080)
y_train = np.zeros(780)
j = 0
for i in y_tr:
    y_train[j] = np.argmax(i)
    j +=1
X_te, y_te = data.get_all_sequences_in_memory('test', data_type)
X_test = X_te.reshape(192,1080)
y_test = np.zeros(192)
j = 0
for i in y_te:
    #print(np.argmax(i))
    y_test[j] = np.argmax(i)
    j +=1
 # training a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print accuracy
 
# creating a confusion matrix
knn_predictions = knn.predict(X_test) 
cm = confusion_matrix(y_test, knn_predictions)
class_names = [1,2,3,4,5,6]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()   