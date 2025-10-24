"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q1. Decision Trees.
"""
# Dominika PIECHOTA & Pawel DOROSZ

import os
import numpy as np
import matplotlib
import random

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data import make_dataset1
from sklearn.tree import DecisionTreeClassifier, plot_tree
from plot import plot_boundary

np.random.seed(42)
random.seed(42)
matplotlib.use('Agg') # no GUI - files saving

# CREATE AND TRAIN A MODEL FOR EACH GIVEN DE
def train_models(train_X, train_y, depths):
    models = [DecisionTreeClassifier(max_depth=d) for d in depths]
    for model in models:
        model.fit(train_X, train_y)
    return models


# RETURN RESULTS FOR EACH MODEL (TRAINING AND TEST ACCURACY)
def evaluate_models(models, train_X, train_y, test_X, test_y):
    results = []
    for model in models:
        results.append({
            "depth": model.max_depth,
            "train_acc": model.score(train_X, train_y),
            "test_acc": model.score(test_X, test_y)
        })
    return results


# SAVE A TREE PLOT FOR EACH MODEL
def save_tree_plots(pdf_filename, models, results):
    with PdfPages(pdf_filename) as pdf:
        for model, res in zip(models, results):
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(model, filled=True, ax=ax)

            ax.set_title(
                f"Decision Tree (max_depth={res['depth']})\n"
                f"Train acc={res['train_acc']:.3f}, Test acc={res['test_acc']:.3f}"
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# SAVE A DECISION BOUNDARY PLOT TO THE FOLDER FOR EACH MODEL
def save_boundary_plots(folder, dataset_id, models, test_X, test_y, results):
    os.makedirs(folder, exist_ok=True)
    for model, res in zip(models, results):
        boundary_path = os.path.join(
            folder, f"dataset{dataset_id}_max_depth={model.max_depth}_dt.pdf"
        )
        plot_boundary(boundary_path[:-4], model, test_X, test_y, title = f"Train acc= {res['train_acc']:.3f}, Test acc={res['test_acc']:.3f}, max_depth={model.max_depth}")


# HANDLE THE ENTIRE PROCESS - DATA GENERATION, TRAINING, PLOTTING
def process_dataset(dataset_id, depths):
    print(f"\n=== DATASET {dataset_id} ===")

    data_X, data_y = make_dataset1(1200, random_state=dataset_id + 41) # manually change the random state to ensure each dataset is different
    train_X, test_X = data_X[:900], data_X[900:]
    train_y, test_y = data_y[:900], data_y[900:]

    # training and evaluation
    models = train_models(train_X, train_y, depths)
    results = evaluate_models(models, train_X, train_y, test_X, test_y)

    # save plots
    pdf_filename = f"dataset_{dataset_id}_trees.pdf"
    boundary_folder = f"dataset_{dataset_id}_boundaries_decision_trees"
    save_tree_plots(pdf_filename, models, results)
    save_boundary_plots(boundary_folder, dataset_id, models, test_X, test_y, results)

    # save training  and test accuracies for the dataset
    train_accs = [r["train_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    print("\nDepth | Train acc | Test acc")
    print("-" * 30)
    for d, tr, te in zip(depths, train_accs, test_accs):
        d_str = f"{d}" if d is not None else "None"
        print(f"{d_str:5} | {tr:9.4f} | {te:8.4f}")

    print(f"✅ Tree plot PDF saved to: {pdf_filename}")
    print(f"📂 Boundary plot saved to: {boundary_folder}")

    return results


if __name__ == "__main__":

    depths = [1, 2, 4, 6, None]
    all_results = []

    for dataset_id in range(1, 6): # 5 generations of the synthetic dataset
        dataset_results = process_dataset(dataset_id, depths)
        all_results.append(dataset_results)

    train_acc_matrix = np.array([[r["train_acc"] for r in res] for res in all_results]) # rows correspond to datasets, columns to max_depths
    test_acc_matrix = np.array([[r["test_acc"] for r in res] for res in all_results])

    # means and standard deviations for each max_depth (both on training and test sets)
    mean_train_acc = np.mean(train_acc_matrix, axis=0)
    std_train_acc = np.std(train_acc_matrix, axis=0)
    mean_test_acc = np.mean(test_acc_matrix, axis=0)
    std_test_acc = np.std(test_acc_matrix, axis=0)

    print("\n=== AVERAGE RESULTS FOR EACH max_depth (mean over 5 synthetic dataset generations) ===")
    print("max_depth\tTrain acc (±std)\t\tTest acc (±std)")
    print("-" * 55)
    for d, mtr, str_, mte, ste in zip(depths, mean_train_acc, std_train_acc, mean_test_acc, std_test_acc):
        d_str = str(d) if d is not None else "None"
        print(f"{d_str:<13}{mtr:.3f} ± {str_:.3f}\t\t{mte:.3f} ± {ste:.3f}")

    # plot average accuracy as a function of k
    depths_labels = [1, 2, 4, 6]  # skip None (hard to contain in the plot)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        depths_labels, mean_train_acc[:-1], yerr=std_train_acc[:-1],
        fmt='-o', capsize=5, label="Train Acc"
    )
    plt.errorbar(
        depths_labels, mean_test_acc[:-1], yerr=std_test_acc[:-1],
        fmt='-s', capsize=5, label="Test Acc"
    )
    plt.xlabel("depth")
    plt.ylabel("Average Accuracy")
    plt.title("Average Train/Test Accuracy Rate (± std) over 5 Random Datasets (dt)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("average_train_test_acc_dt.pdf")
    plt.close()

    print("\n📊 Plot saved as average_train_test_acc_dt.pdf")