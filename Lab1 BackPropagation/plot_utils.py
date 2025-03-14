import matplotlib.pyplot as plt
import numpy as np

def plot_epoch_acc_curve(acc_list,type):
    plt.figure()
    plt.plot(np.arange(len(acc_list)), acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig("output/acc_curve_{}.png".format(type))
    plt.show()

def plot_loss_curve(losses,type):
    plt.figure()
    plt.plot(np.arange(len(losses)), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig("output/loss_curve_{}.png".format(type))
    plt.show()

def show_result(type,x, y, pred_y,save_fig = True):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    if save_fig:
        plt.savefig("output/prediction_{}.png".format(type))    
    plt.show()
