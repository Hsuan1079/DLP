import numpy as np
import matplotlib.pyplot as plt
from model import NN
from plot_utils import show_result
from data import load_data
from main import train

def experiment_learning_rates(learning_rates, epochs=100000, data_type='linear'):

    x_train, y_train = load_data(data_type)
    x_test, y_test = x_train, y_train

    results = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")

        model = NN(x_train.shape[1], 10, 10, 1, act_type='relu',optimizer='sgd')

        losses, acc_list = train(model, x_train, y_train, epochs, lr)

        predictions = model.forward(x_test)
        preds_label = (predictions > 0.5).astype(int)
        show_result(f"{data_type}_lr_{lr}", x_test, y_test, preds_label,save_fig = False)

        results[lr] = (losses, acc_list)
    plot_results(results, "Learning Rates", data_type)

    return results

def experiment_hidden_units(hidden_units_list, epochs=100000, data_type='linear'):
    x_train, y_train = load_data(data_type)
    x_test, y_test = x_train, y_train

    results = {}

    for hidden_units in hidden_units_list:
        print(f"\nTraining with hidden units: {hidden_units}")

        model = NN(x_train.shape[1], hidden_units, hidden_units, 1, act_type='relu',opttimizer='sgd')

        losses, acc_list = train(model, x_train, y_train, epochs, 0.001)

        predictions = model.forward(x_test)
        preds_label = (predictions > 0.5).astype(int)
        show_result(f"{data_type}_hidden_{hidden_units}", x_test, y_test, preds_label,save_fig = False)

        results[hidden_units] = (losses, acc_list)
    plot_results(results, "Hidden Units", data_type)

    return results

def experiment_no_activation(epochs=100000, data_type='linear'):
    x_train, y_train = load_data(data_type)
    x_test, y_test = x_train, y_train

    print(f"\nTraining without Activation Function")

    model = NN(x_train.shape[1], 10, 10, 1, act_type='none',optimizer='sgd')

    losses, acc_list = train(model, x_train, y_train, epochs, 0.001)

    predictions = model.forward(x_test)
    preds_label = (predictions > 0.5).astype(int)
    show_result(f"{data_type}_no_activation_hidden_10", x_test, y_test, preds_label,save_fig = False)

    return losses, acc_list

def experiment_activation_functions(activations, epochs=100000, data_type='linear'):
    x_train, y_train = load_data(data_type)
    x_test, y_test = x_train, y_train

    results = {}

    for activation in activations:
        print(f"\nTraining with Activation Function: {activation}")

        model = NN(x_train.shape[1], 10, 10, 1, act_type=activation,optimizer='sgd')

        losses, acc_list = train(model, x_train, y_train, epochs, 0.001)

        predictions = model.forward(x_test)
        preds_label = (predictions > 0.5).astype(int)
        show_result(f"{data_type}_activation_{activation}", x_test, y_test, preds_label, save_fig=False)

        results[activation] = (losses, acc_list)

    return results

def experiment_optimizers(optimizer_list, epochs=100000, data_type='linear'):
    x_train, y_train = load_data(data_type)
    x_test, y_test = x_train, y_train

    results = {}

    for optimizer in optimizer_list:
        print(f"\nTraining with optimizer: {optimizer}")

        model = NN(x_train.shape[1], 10, 10, 1, act_type='relu', optimizer=optimizer)

        losses, acc_list = train(model, x_train, y_train, epochs, 0.01)

        predictions = model.forward(x_test)
        preds_label = (predictions > 0.5).astype(int)
        show_result(f"{data_type}_optimizer_{optimizer}", x_test, y_test, preds_label, save_fig=False)

        results[optimizer] = (losses, acc_list)

    plot_optimizer_results(results, "Optimizer Comparison", data_type)

    return results

def plot_results(results, test_type, data_type):
    plt.figure(figsize=(12, 5))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    for key, (losses, acc_list) in results.items():
        plt.plot(range(len(acc_list)), acc_list, label=f"{test_type} {key}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison ({test_type})')
    plt.legend()
    plt.grid(True)

    # Loss comparison
    plt.subplot(1, 2, 2)
    for key, (losses, acc_list) in results.items():
        plt.plot(range(len(losses)), losses, label=f"{test_type} {key}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison ({test_type})')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"output/{test_type.replace(' ', '_').lower()}_comparison_{data_type}.png")
    plt.show()

def plot_results_with_no_activation(results_relu, losses_none, acc_none, data_type):
    # compare with and without activation
    plt.figure(figsize=(12, 5))

    # Accuracy 比較
    plt.subplot(1, 2, 1)
    for key, (losses, acc_list) in results_relu.items():
        if key == 10: 
            plt.plot(range(len(acc_list)), acc_list, label=f"ReLU (Hidden 10)", linestyle='solid')
    plt.plot(range(len(acc_none)), acc_none, label="No Activation (Hidden 10)", linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison (w/ and w/o Activation)')
    plt.legend()
    plt.grid(True)

    # Loss 
    plt.subplot(1, 2, 2)
    for key, (losses, acc_list) in results_relu.items():
        if key == 10:  
            plt.plot(range(len(losses)), losses, label=f"ReLU (Hidden 10)", linestyle='solid')
    plt.plot(range(len(losses_none)), losses_none, label="No Activation (Hidden 10)", linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison (w/ and w/o Activation)')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"output/activation_comparison_{data_type}.png")
    plt.show()

def plot_results_with_activations(results, data_type):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for activation, (losses, acc_list) in results.items():
        plt.plot(range(len(acc_list)), acc_list, label=f"{activation}", linestyle='solid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison (Different Activation Functions)')
    plt.legend()
    plt.grid(True)

    # Loss 比較
    plt.subplot(1, 2, 2)
    for activation, (losses, acc_list) in results.items():
        plt.plot(range(len(losses)), losses, label=f"{activation}", linestyle='solid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison (Different Activation Functions)')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"output/activation_functions_comparison_{data_type}.png")
    plt.show()

def plot_optimizer_results(results, test_type, data_type):
    plt.figure(figsize=(12, 5))

    # Accuracy 比較
    plt.subplot(1, 2, 1)
    for optimizer, (losses, acc_list) in results.items():
        plt.plot(range(len(acc_list)), acc_list, label=f"{optimizer}", linestyle='solid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison ({test_type})')
    plt.legend()
    plt.grid(True)

    # Loss 比較
    plt.subplot(1, 2, 2)
    for optimizer, (losses, acc_list) in results.items():
        plt.plot(range(len(losses)), losses, label=f"{optimizer}", linestyle='solid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison ({test_type})')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"output/optimizer_comparison_{data_type}.png")
    plt.show()


if __name__ == "__main__":
    data_type = 'xor'  # 或 'linear'
    
    # # A. test different learning rates 
    # learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    # experiment_learning_rates(learning_rates, data_type=data_type)

    # # B. test different hidden units
    # hidden_units_list = [5, 10, 20, 50]
    # experiment_hidden_units(hidden_units_list, data_type=data_type)

    #  C. test w/ and w/o Activation
    # hidden_units_list = [10]
    # results_relu = experiment_hidden_units(hidden_units_list, data_type=data_type)
    
    # losses_none, acc_none = experiment_no_activation(data_type=data_type)
    # # (ReLU vs No Activation)
    # plot_results_with_no_activation(results_relu, losses_none, acc_none, data_type)

    # test different optimizer
    optimizer_list = ['sgd', 'momentum']
    results_optimizers = experiment_optimizers(optimizer_list, data_type=data_type)


    # # test different activation functions
    # activation_list = ['relu', 'sigmoid', 'tanh', 'none']
    # results_activations = experiment_activation_functions(activation_list, data_type=data_type)

    # plot_results_with_activations(results_activations, data_type)
