from data import load_data
from model import NN
import numpy as np
from plot_utils import show_result, plot_loss_curve, plot_epoch_acc_curve

# np.random.seed(42) 

def compute_loss(y_pred,y_true):
    # MSE loss
    loss = np.mean((y_pred - y_true)**2)
    return loss

def compute_accuracy(y_true,y_pred):
    acc = np.sum(np.equal(y_true,y_pred))/y_pred.shape[0]
    return acc

def train(model,x,y,epochs,learning_rate):
    losses = []
    acc_list = []
    # return losses
    for epoch in range(epochs):
        # forward pass
        y_pred = model.forward(x)
        
        # calculate loss
        loss = compute_loss(y_pred,y)
        losses.append(loss)

        # calulare accuracy
        y_pred_labels = (y_pred > 0.5).astype(int)  
        acc = compute_accuracy(y, y_pred_labels)  
        acc_list.append(acc)

        if acc == 1.0:
            print(f"Training stopped early at epoch {epoch+1} with 100% accuracy\n")
            break

        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"epoch {epoch+1:<5} loss : {loss:.16f}, acc : {acc:.4f}")
        
        # backward pass
        gradients = model.backward(x,y)
        model.update(gradients,learning_rate)
    
    return losses, acc_list

def test(model,x_test,y_test):
    predictions = model.forward(x_test)
    preds_label = (predictions > 0.5).astype(int)
    acc = compute_accuracy(y_test, preds_label)
    loss = compute_loss(predictions, y_test)

    for i in range(len(y_test)):
        print(f"Iter{i+1:02}  |  Ground truth: {y_test[i][0]:.1f}  |  prediction: {predictions[i][0]:.5f}")
    
    print(f"\nloss={loss:.5f} accuracy={acc * 100:.2f}%\n")
    return preds_label

if __name__ == "__main__":
    epochs = 100000
    learning_rate = 0.001
    type = 'linear' 
    # type = 'xor'

    # load data
    x_train, y_train = load_data(type)
    x_test, y_test = x_train, y_train

    # initialize model
    d_in = x_train.shape[1]
    d_hidden1 = 20 
    d_hidden2 = 20
    d_out = 1
    
    model = NN(d_in,d_hidden1,d_hidden2,d_out,act_type='relu',optimizer='sgd')

    # train model
    losses, acc_list = train(model,x_train,y_train,epochs,learning_rate)

    # test model
    prediction= test(model,x_test,y_test)

    # plot
    plot_epoch_acc_curve(acc_list,type)
    plot_loss_curve(losses,type)
    show_result(type,x_test,y_test,prediction)


