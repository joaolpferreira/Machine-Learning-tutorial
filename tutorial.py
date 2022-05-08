import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nn_utils
import pandas as pd
from keras.datasets import mnist  
from tensorflow import keras 
from keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D, Concatenate, LeakyReLU
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, Reshape, Input
from tensorflow.keras.optimizers import SGD

class Tensor:
    def load_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
        x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        return (x_train, y_train), (x_test, y_test)
    
    def plot_results(history):
        plt.figure(figsize=(12, 4))
        epochs = len(history['val_loss'])
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), history['val_loss'], label='Val Loss')
        plt.plot(range(epochs), history['train_loss'], label='Train Loss')
        plt.xticks(list(range(epochs)))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), history['val_acc'], label='Val Acc')
        plt.xticks(list(range(epochs)))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        return plt

    class NeuralNetwork:
        def __init__(self, layers):
            self.layers = layers
            self.L = len(layers)
            self.num_features = layers[0]
            self.num_classes = layers[-1]
            
            self.W = {}
            self.b = {}
            
            self.dW = {}
            self.db = {}
            
            self.setup()
            
        def setup(self):      
            for i in range(1, self.L):
                self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i],self.layers[i-1])))
                self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i],1)))
                
        def forward_pass(self, X): 
            A = tf.convert_to_tensor(X, dtype=tf.float32)
            for i in range(1, self.L):
                Z = tf.matmul(A,tf.transpose(self.W[i])) + tf.transpose(self.b[i])
                if i != self.L-1:
                    A = tf.nn.relu(Z)
                else:
                    A = Z
            return A
           
        def compute_loss(self, A, Y):
            loss = tf.nn.softmax_cross_entropy_with_logits(Y,A)
            return tf.reduce_mean(loss)
        
        def update_params(self, lr):
            for i in range(1,self.L):
                self.W[i].assign_sub(lr * self.dW[i])
                self.b[i].assign_sub(lr * self.db[i])
                    
        def predict(self, X):  
            A = self.forward_pass(X)
            return tf.argmax(tf.nn.softmax(A), axis=1)
        
        def info(self):
            num_params = 0
            for i in range(1, self.L):
                num_params += self.W[i].shape[0] * self.W[i].shape[1]
                num_params += self.b[i].shape[0]  
            print('Number of parameters:', num_params)
            
        def train_on_batch(self, X, Y, lr):             
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            Y = tf.convert_to_tensor(Y, dtype=tf.float32)             
            with tf.GradientTape(persistent=True) as tape:
                A = self.forward_pass(X)
                loss = self.compute_loss(A, Y)
            for i in range(1, self.L):
                self.dW[i] = tape.gradient(loss, self.W[i])
                self.db[i] = tape.gradient(loss, self.b[i])
            del tape
            self.update_params(lr)
            return loss.numpy()
         
        def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):  
            history = {
                'val_loss':[],
                'train_loss':[],
                'val_acc':[]
            }           
            for e in range(0, epochs):
                epoch_train_loss = 0.
                print('Epoch{}'.format(e), end='.')
                for i in range(0, steps_per_epoch):
                    x_batch = x_train[i*batch_size:(i+1)*batch_size]
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]
                    
                    batch_loss = self.train_on_batch(x_batch, y_batch,lr)
                    epoch_train_loss += batch_loss
                    
                    if i%int(steps_per_epoch/10) == 0:
                        print(end='.')
                        
                history['train_loss'].append(epoch_train_loss/steps_per_epoch)
                val_A = self.forward_pass(x_test)
                val_loss = self.compute_loss(val_A, y_test).numpy()
                #print(np.exp(val_loss)/np.exp(val_A))
                history['val_loss'].append(val_loss)
                val_preds = self.predict(x_test)
                val_acc =    np.mean(np.argmax(y_test, axis=1) == val_preds.numpy())
                history['val_acc'].append(val_acc)               
                print('Val acc:',val_acc)
            return history
        
    (x_train, y_train), (x_test, y_test) = load_data()   
    
    #Create Neural Network
    net = NeuralNetwork([784,512,256,10])
    #Display Number of Parameters
    net.info()
    
    #Hyperparameters
    batch_size = 64
    epochs = 11
    steps_per_epoch = int(x_train.shape[0]/batch_size)
    lr = 0.005
    print('Steps per epoch', steps_per_epoch)
    
    #Trainning and Validation
    history = net.train(
        x_train,y_train,
        x_test, y_test,
        epochs, steps_per_epoch,
        batch_size, lr)
    
    #Graph
    plot_results(history).show()
    
class Keras_Sequential_Mnist:
    
    #Data Selection
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    
    X_valid, X_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    
    #Just to fix tensorflow and tensorflow-gpu conflict in my computer
    with tf.device('/CPU:0'):
        
        model = keras.models.Sequential()
        
    #Structure of the model
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        
    #Summary
        model.summary()
        
    #Image with model summary
        plot_model(model, to_file='model_Seq_Mnist.png', show_shapes=True)
        
    #Optimizer and compile
        INIT_LR = 0.01
        NUM_EPOCHS = 11
        opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        
    #Trainning and Validation
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, validation_data=(X_valid, y_valid))
    
    #Graph
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) 
        plt.show()

class Keras_Functional_Mnist:
    #Get Data
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    X_valid, X_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    with tf.device('/CPU:0'):
        model = keras.models.Sequential()
        
        #Structure
        input_      =   Input(shape=[28, 28,1])
        hidden01    =   Conv2D(64,kernel_size=3, activation='relu') (input_)
        hidden_     =   BatchNormalization()(hidden01)
        hidden02    =   Conv2D(64,kernel_size=3, activation='relu') (hidden_)
        hidden_1    =   BatchNormalization()(hidden02)
        hidden03    =   MaxPooling2D(pool_size=(2))(hidden_1)
        hidden04    =   Conv2D(64,kernel_size=3, activation='relu') (hidden03)
        hidden_2    =   BatchNormalization()(hidden04)
        hidden05    =   MaxPooling2D(pool_size=(2))(hidden_2)
        flatten     =   Flatten(input_shape=[28, 28])(hidden05)
        hidden06    =   Dense((784), activation='relu')(flatten)
        hidden_3    =   BatchNormalization()(hidden06)
        reshap      =   Reshape((28, 28,1))(hidden_3)
        
        concat_     =   Concatenate()([input_, reshap])
        flatten2    =   Flatten(input_shape=[28, 28,1])(concat_)
        drop        =   Dropout(0.5)(flatten2)
        output      =   Dense(10, activation='softmax')(drop)
        
        model = keras.Model(inputs=[input_], outputs=[output] )
        
        #Summary
        model.summary()
        
        #Image with struct
        plot_model(model, to_file='model_Func_Mnist.png')
        
        #Optimizer
        INIT_LR = 0.01
        NUM_EPOCHS = 11
        opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        
        #Graph
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, validation_data=(X_valid, y_valid))
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) 
        plt.show()

class Keras_Sequential_FashionMnist:
    ((trainX, trainY), (testX, testY)) = keras.datasets.fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    # one-hot encode the training and testing labels
    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)

    with tf.device('/CPU:0'):
       model = keras.models.Sequential()
       
       #Structure
       model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
       model.add(BatchNormalization())
       
       model.add(Conv2D(64, (3, 3), activation='relu'))
       model.add(BatchNormalization())
       
       model.add(MaxPooling2D(pool_size=(2, 2)))
       
       model.add(Conv2D(64, (3, 3), activation='relu'))
       model.add(BatchNormalization())
       
       model.add(MaxPooling2D(pool_size=(2, 2)))
       
       model.add(Flatten())
       model.add(Dense(512, activation='relu'))
       model.add(BatchNormalization())
       model.add(Dropout(0.5))
       model.add(Dense(10))
       model.add(Activation("softmax"))
       
       #Summary
       model.summary()
       
       #Image with struct
       plot_model(model, to_file='model_Seq_Fashion.png')
       
       #Optimizer and compile
       INIT_LR = 0.01
       NUM_EPOCHS = 11
       opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
       model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
       
       ##Trainning and Validation
       history = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=NUM_EPOCHS)
       
       #Graph
       pd.DataFrame(history.history).plot(figsize=(8, 5))
       plt.grid(True)
       plt.gca().set_ylim(0, 1) 
       plt.show()
       
class Keras_Functional_FashionMnist:
    #Data Selection
    ((trainX, trainY), (testX, testY)) = keras.datasets.fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)

    #Just to fix tensorflow and tensorflow-gpu conflict in my computer
    with tf.device('/CPU:0'):
        model = keras.models.Sequential()
        
    #Structure
        input_      =   Input(shape=[28, 28,1])
        hidden01    =   Conv2D(64,kernel_size=3, activation='relu') (input_)
        hidden_     =   BatchNormalization()(hidden01)
        hidden02    =   Conv2D(64,kernel_size=3, activation='relu') (hidden_)
        hidden_1    =   BatchNormalization()(hidden02)
        hidden03    =   MaxPooling2D(pool_size=(2))(hidden_1)
        hidden04    =   Conv2D(64,kernel_size=3, activation='relu') (hidden03)
        hidden_2    =   BatchNormalization()(hidden04)
        hidden05    =   MaxPooling2D(pool_size=(2))(hidden_2)
        flatten     =   Flatten(input_shape=[28, 28])(hidden05)
        hidden06    =   Dense((784), activation='relu')(flatten)
        hidden_3    =   BatchNormalization()(hidden06)
        reshap      =   Reshape((28, 28,1))(hidden_3)    
        concat_     =   Concatenate()([input_, reshap])
        flatten2    =   Flatten(input_shape=[28, 28,1])(concat_)
        drop        =   Dropout(0.5)(flatten2)
        output      =   Dense(10, activation='softmax')(drop)   
        model = keras.Model(inputs=[input_], outputs=[output] )
        
    #Summary
        model.summary()
        
    #Image with model summary
        plot_model(model, to_file='model_Func_Fashion.png', show_shapes=True)
        
    #Optimizer and compile
        INIT_LR = 0.02
        NUM_EPOCHS = 11  
        opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)    
        model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        
    #Trainning and Validation
        history = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=NUM_EPOCHS)

    #Graph    
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) 
        plt.show()
        
class Keras_ComplexFunctional_FashionMnist:
    #Data Selection
    ((trainX, trainY), (testX, testY)) = keras.datasets.fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)
    
    #Just to fix tensorflow and tensorflow-gpu conflict in my computer
    with tf.device('/CPU:0'):
        model = keras.models.Sequential()
        
        #Structure
        input_      =   Input(shape=[28, 28,1])
        hidden01    =   Conv2D(128,kernel_size=3) (input_)
        hidden14    =   LeakyReLU(0.5)(hidden01)
        hidden_     =   BatchNormalization()(hidden14)
        hidden02    =   Conv2D(128,kernel_size=3) (hidden_)
        hidden15    =   LeakyReLU(0.5)(hidden02)
        hidden_1    =   BatchNormalization()(hidden15)
        hidden03    =   MaxPooling2D(pool_size=(2))(hidden_1)
        hidden04    =   Conv2D(128,kernel_size=3) (hidden03)
        hidden16    =   LeakyReLU(0.5)(hidden04)
        hidden_2    =   BatchNormalization()(hidden16)
        hidden05    =   MaxPooling2D(pool_size=(2))(hidden_2)
        flatten     =   Flatten(input_shape=[28, 28])(hidden05)
        hidden06    =   Dense((784))(flatten)
        hidden17    =   LeakyReLU(0.5)(hidden06)
        hidden_3    =   BatchNormalization()(hidden17)
        reshap      =   Reshape((28, 28,1))(hidden_3)
        
        hidden07    =   Conv2D(128,kernel_size=4, padding="same") (input_)
        hidden08    =   LeakyReLU(0.5)(hidden07)
        hidden09    =   BatchNormalization()(hidden08)
        hidden10    =   MaxPooling2D(pool_size=(2))(hidden09)
        flatten2    =   Flatten(input_shape=[28, 28,1])(hidden10)
        hidden11    =   Dense(784)(flatten2)
        hidden13    =   LeakyReLU(0.5)(hidden11)
        hidden12    =   BatchNormalization()(hidden13)
        reshap1     =   Reshape((28,28,1))(hidden12)
              
        concat_     =   Concatenate()([input_, reshap])
        
        concat_1    =   Concatenate()([concat_,reshap1])
        flatten4    =   Flatten(input_shape=[28, 28,1])(concat_1)
        
        drop        =   Dropout(0.5)(flatten4)
        output      =   Dense(10, activation='softmax')(drop)
        
        model = keras.Model(inputs=[input_], outputs=[output] )
        
        #Summary
        model.summary()
        
        #Image with struct
        plot_model(model, to_file='model_Func_Fashion_Best.png')
        
        #Optimizer and compile
        INIT_LR = 0.01
        NUM_EPOCHS = 11
        opt = SGD(learning_rate=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        #Trainning and Validation
        history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=NUM_EPOCHS)
        
        #Graph
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) 
        plt.show()