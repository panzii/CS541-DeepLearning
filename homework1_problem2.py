import numpy as np

hyperparameter_configs = [
    # [learning_rate, number_of_epochs, mini_batch_size]
    [0.0001, 50, 16],
    [0.0001, 50, 32],
    [0.0001, 100, 16],
    [0.0001, 100, 32],
    [0.001, 50, 16],
    [0.001, 50, 32],
    [0.001, 100, 16],
    [0.001, 100, 32],
    [0.001, 200, 16],
    [0.001, 200, 32],
]

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.reshape(np.load("age_regression_ytr.npy"), (-1, 1))
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.reshape(np.load("age_regression_yte.npy"), (-1, 1))
    
    # Appended 1.0 at the end of each image vector for the bias term
    X_tr = np.concatenate((X_tr, np.ones((X_tr.shape[0], 1))), axis=1)
    X_te = np.concatenate((X_te, np.ones((X_te.shape[0], 1))), axis=1)

    # spliting training examples into validation and training set
    X_val = X_tr[4000:]
    yval = ytr[4000:]

    X_tr = X_tr[0:4000]
    ytr = ytr[0:4000]

    # starting with arbitrary weights
    n = X_tr.shape[0]
    m = X_tr.shape[1]
    W = np.random.rand(m, 1)

    results = [] # List[List] => [config, W, validation_loss]
    for config in hyperparameter_configs:
        mini_batch_size = config[2]
        no_of_epochs = config[1]
        learning_rate = config[0]

        shuffle_indices = np.random.permutation(X_tr.shape[0])
        X_tr = X_tr[shuffle_indices]
        ytr = ytr[shuffle_indices]

        for epoch in range(no_of_epochs):
            for i in range(0, n, mini_batch_size):
                # getting the minibatch initialized
                X = X_tr[i:i+mini_batch_size]
                Y = ytr[i:i+mini_batch_size]

                y_hat = np.dot(X, W)

                # calculating gradient
                del_f_MSE = (1/mini_batch_size)*np.dot(X.T, y_hat-Y)

                # updating weights
                W = W - (learning_rate*del_f_MSE)

            training_loss = (np.sum(np.square(np.dot(X_tr, W) - ytr))*(1/(2*n)))       

        validation_loss = (np.sum(np.square(np.dot(X_val, W) - yval))*(1/(2*X_val.shape[0])))
        print(f"config: {config}, validation loss: {validation_loss}, training loss: {training_loss}")
        results.append([config, W, validation_loss])
    
    # getting weights and hyperparameter config which produced minimum validation loss 
    min_val_loss_config = min(results, key=lambda x: x[2])
    W = min_val_loss_config[1]

    training_loss = (np.sum(np.square(np.dot(X_tr, W) - ytr))*(1/(2*n)))
    print(f"Training loss: {training_loss}")
    testing_loss = (np.sum(np.square(np.dot(X_te, W) - yte))*(1/(2*X_te.shape[0])))
    print("Testing Loss: ", testing_loss)

train_age_regressor()