import numpy as np

class activation:
    def __init__(self,act_type):
        self.act_type = act_type

    def forward(self,x):
        if self.act_type == 'relu':
            return np.maximum(0,x)
        elif self.act_type == 'sigmoid':
            return 1.0/(1.0+np.exp(-x))
        elif self.act_type == 'tanh':
            return np.tanh(x)
        elif self.act_type == 'none':
            return x
    
    def derivative(self,x):
        if self.act_type == 'relu':
            return (x > 0).astype(float) 
        elif self.act_type == 'sigmoid':
            return np.multiply(x,1-x)
        elif self.act_type == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.act_type == 'none':
            return np.ones_like(x) 

class NN:
    def __init__(self,d_in,d_hidden1,d_hidden2,d_out,act_type,optimizer='sgd'):
        # random initialization of weights
        self.W1 = np.random.randn(d_in,d_hidden1)
        self.b1 = np.zeros((1,d_hidden1))

        self.W2 = np.random.randn(d_hidden1,d_hidden2)
        self.b2 = np.zeros((1,d_hidden2))

        self.W3 = np.random.randn(d_hidden2,d_out)
        self.b3 = np.zeros((1,d_out))
        
        self.act = activation(act_type)

        self.out_act = activation('sigmoid')
        self.optimizer = optimizer

        if optimizer == 'momentum':
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)
            self.v_W3 = np.zeros_like(self.W3)
            self.v_b3 = np.zeros_like(self.b3)
            self.beta = 0.9  
        

    def forward(self,x):
        # z = act(Wx + b)
        self.z1 = np.dot(x,self.W1) + self.b1
        self.a1 = self.act.forward(self.z1)

        self.z2 = np.dot(self.a1,self.W2) + self.b2
        self.a2 = self.act.forward(self.z2)

        self.z3 = np.dot(self.a2,self.W3) + self.b3
        self.a3 = self.out_act.forward(self.z3)

        return self.a3
    
    def backward(self,x,y):
        m = x.shape[0]
        dz3 = (self.a3 - y)*self.out_act.derivative(self.a3)
        dW3 = np.dot(self.a2.T,dz3)/m
        db3 = np.sum(dz3,axis=0,keepdims=True)/m

        dz2 = np.dot(dz3,self.W3.T)*self.act.derivative(self.a2)
        dW2 = np.dot(self.a1.T,dz2)/m
        db2 = np.sum(dz2,axis=0,keepdims=True)/m

        dz1 = np.dot(dz2,self.W2.T)*self.act.derivative(self.a1)
        dW1 = np.dot(x.T,dz1)/m
        db1 = np.sum(dz1,axis=0,keepdims=True)/m

        gradients = [dW1,db1,dW2,db2,dW3,db3]
        return gradients
    
    def update(self,gradients,learning_rate):
        dW1, db1, dW2, db2, dW3, db3 = gradients

        if self.optimizer == 'sgd':
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
        elif self.optimizer == 'momentum':
            self.v_W1 = self.beta * self.v_W1 + (1 - self.beta) * dW1
            self.v_b1 = self.beta * self.v_b1 + (1 - self.beta) * db1
            self.v_W2 = self.beta * self.v_W2 + (1 - self.beta) * dW2
            self.v_b2 = self.beta * self.v_b2 + (1 - self.beta) * db2
            self.v_W3 = self.beta * self.v_W3 + (1 - self.beta) * dW3
            self.v_b3 = self.beta * self.v_b3 + (1 - self.beta) * db3

            self.W1 -= learning_rate * self.v_W1
            self.b1 -= learning_rate * self.v_b1
            self.W2 -= learning_rate * self.v_W2
            self.b2 -= learning_rate * self.v_b2
            self.W3 -= learning_rate * self.v_W3
            self.b3 -= learning_rate * self.v_b3
