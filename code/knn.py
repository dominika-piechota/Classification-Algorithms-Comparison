"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q2. k-Nearest Neighbors (kNN).
"""
# Dominika PIECHOTA & Pawel DOROSZ

import os
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from data import make_dataset1
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary

matplotlib.use('Agg')  # no GUI - files saving

# CREATE AND TRAIN A MODEL FOR EACH GIVEN K
def train_models(train_X, train_y, num_of_neighbors):
    models = [KNeighborsClassifier(n_neighbors=n) for n in num_of_neighbors]
    for model in models:
        model.fit(train_X, train_y)
    return models


# RETURN RESULTS FOR EACH MODEL (TRAINING AND TEST ACCURACY)
def evaluate_models(models, train_X, train_y, test_X, test_y):
    results = []
    for model in models:
        results.append({
            "k": model.n_neighbors,
            "train_acc": model.score(train_X, train_y),
            "test_acc": model.score(test_X, test_y)
        })
    return results


# SAVE A DECISION BOUNDARY PLOT TO THE FOLDER FOR EACH MODEL
def save_boundary_plots(folder, dataset_id, models, test_X, test_y, results):
    os.makedirs(folder, exist_ok=True)
    for model, res in zip(models, results):
        boundary_path = os.path.join(
            folder, f"dataset{dataset_id}_k={model.n_neighbors}.pdf"
        )
        plot_boundary(boundary_path[:-4], model, test_X, test_y, title = f"Train acc= {res['train_acc']:.3f}, Test acc={res['test_acc']:.3f}, k={model.n_neighbors}")


# HANDLE THE ENTIRE PROCESS - DATA GENERATION, TRAINING, PLOTTING
def process_dataset(dataset_id, num_of_neighbours):
    print(f"\n=== DATASET {dataset_id} ===")

    data_X, data_y = make_dataset1(1200, random_state=dataset_id+41) # manually change the random state to ensure each dataset is different
    train_X, test_X = data_X[:900], data_X[900:] # 1200 * 0.75 = 900
    train_y, test_y = data_y[:900], data_y[900:]

    # training and evaluation
    models = train_models(train_X, train_y, num_of_neighbours)
    results = evaluate_models(models, train_X, train_y, test_X, test_y)

    # save boundary plots
    boundary_folder = f"dataset_{dataset_id}_boundaries_knn"
    save_boundary_plots(boundary_folder, dataset_id, models, test_X, test_y, results)

    # save training  and test accuracies for the dataset
    train_accs = [r["train_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    print("\nk-value | Train acc | Test acc")
    print("-" * 30)
    for k, tr, te in zip(num_of_neighbours, train_accs, test_accs):
        print(f"{k:5} | {tr:9.4f} | {te:8.4f}")

    print(f"📂 Boundary plot saved to: {boundary_folder}")

    return results


if __name__ == "__main__":

    num_of_neighbours = [1, 5, 25, 125, 500, 899]
    all_results = []

    for dataset_id in range(1,6): # 5 generations of the synthetic dataset
        dataset_results = process_dataset(dataset_id, num_of_neighbours)
        all_results.append(dataset_results)

    train_acc_matrix = np.array([[r["train_acc"] for r in res] for res in all_results]) # rows correspond to datasets, columns to k-values
    test_acc_matrix = np.array([[r["test_acc"] for r in res] for res in all_results])

    # means and standard deviations for each k value (both on training and test sets)
    mean_train_acc = np.mean(train_acc_matrix, axis=0)
    std_train_acc = np.std(train_acc_matrix, axis=0)
    mean_test_acc = np.mean(test_acc_matrix, axis=0)
    std_test_acc = np.std(test_acc_matrix, axis=0)

    print("\n=== AVERAGE RESULTS FOR EACH k (mean over 5 synthetic dataset generations) ===")
    print("k\t    Train acc (±std)\tTest acc (±std)")
    print("-" * 55)
    for k, mtr, str_, mte, ste in zip(num_of_neighbours, mean_train_acc, std_train_acc, mean_test_acc, std_test_acc):
        print(f"{k:<8}{mtr:.3f} ± {str_:.3f}\t\t{mte:.3f} ± {ste:.3f}")

    #plot average accuracy as a function of k
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        num_of_neighbours, mean_train_acc, yerr=std_train_acc,
        fmt='-o', capsize=5, label="Train Acc"
    )
    plt.errorbar(
        num_of_neighbours, mean_test_acc, yerr=std_test_acc,
        fmt='-s', capsize=5, label="Test Acc"
    )
    plt.xscale('log')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Average Accuracy")
    plt.title("Average Train/Test Accuracy Rate (± std) over 5 Random Datasets (kNN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("average_train_test_acc_kNN.pdf")
    plt.close()

    print("\n📊 Plot saved as average_train_test_acc_kNN.pdf")