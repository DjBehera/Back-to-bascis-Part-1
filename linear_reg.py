import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data():
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.drop(['Profit'],axis=1)
    Y = dataset['Profit']
    
    categorical_list = ['State']
    normaliztaion_list = ['R&D Spend', 'Administration', 'Marketing Spend']
    
    for i in categorical_list:
        temp_name = str(X[i][0])
        X = pd.get_dummies(X, columns=[i])
        X = X.drop([i+'_%s' %(temp_name)],axis = 1)
    
    for i in normaliztaion_list:
        X[i] = (X[i] - X[i].mean()) / X[i].std()
    
    
    Y = (Y - Y.mean())/ Y.std() 
    X = X.values
    Y = Y.values.reshape(-1,1)
    
    return X,Y

def compute_cost(y_pred,y):
    '''
        Cost function calculation for Linear Regression:
            Cost Function J = (1/2*m) * sum(y_pred - y)^2
            m = no of samples
    '''
    a = ((y_pred - y)**2) / (len(y)*2)
    return a.sum()

def derivative(y_pred,y,x):
    '''
        Matrix form of Derivative of cost function wrt to the wieghts initialized
    '''
    b = sum((y_pred - y) * x)
    return b

def main():
    learning_rate = 0.005
    J = []
    X,y = get_data()
    w = np.random.randn(X.shape[1])
    for i in range(1000):
        y_pred = X * w
        cost = compute_cost(y_pred,y)
        print('Cost Function::'+str(cost))
        w = w - (learning_rate / len(X)) *(derivative(y_pred,y,X))
        J.append(cost)    
    ##Showing the cost function in Matplot
    plt.figure()
    plt.xlabel('No of epochs')
    plt.ylabel('Cost func')
    plt.plot(J)
    plt.show()

if __name__ == '__main__':
    main()
		
		
