"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q4. Method comparison.
"""
#Dominika PIECHOTA & Pawel DOROSZ

import os
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from data import make_dataset1, make_dataset_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary
from qda import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('Agg')  # # no GUI - files saving


#5-FOLD CROSS-VALIDATION IMPLEMENTATION FOR kNN
def tune_knn(train_X, train_y, candidate_neighbors):
    best_k = None
    best_score = 0
    n_samples = len(train_X)

    kf = KFold(n_splits=5, shuffle=True, random_state=0) # 5 splits => 80% test and 20% validation

    for k in candidate_neighbors:

        if k > n_samples * (4/5):  # skip too big k (like 899)
            continue

        model = KNeighborsClassifier(n_neighbors=k)
        scores = []

        for train_idx, val_idx in kf.split(train_X):
            X_tr, X_val = train_X[train_idx], train_X[val_idx]
            y_tr, y_val = train_y[train_idx], train_y[val_idx]

        model.fit(X_tr, y_tr)
        acc = model.score(X_val, y_val)
        scores.append(acc)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, best_score


# USING CROSS-VALIDATION TO FIND THE BEST max_depth
def tune_decision_tree(train_X, train_y, depths):
    scores = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(model, train_X, train_y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
        mean_score = np.mean(cv_scores)
        scores.append(mean_score)

    best_depth = depths[np.argmax(scores)]
    best_score = np.max(scores)
    return best_depth, best_score

# WHOLE PROCESS HANDLING - DATA GENERATION, TRAINING, PLOTS
def process_dataset(dataset_id, num_of_neighbours, depths):
    print(f"\n===SYNTHETIC DATASET {dataset_id}===")

    data_X, data_y = make_dataset1(1200, random_state=dataset_id+41) # manually change the random state to ensure each dataset is different
    train_X, test_X = data_X[:900], data_X[900:] # 1200 * 0.75 = 900
    train_y, test_y = data_y[:900], data_y[900:]

    # tuning k
    best_k, best_k_score = tune_knn(train_X, train_y, num_of_neighbours)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(train_X, train_y)

    results_tuned_knn={
        "k": best_knn.n_neighbors,
        "train_acc": best_knn.score(train_X, train_y),
        "test_acc": best_knn.score(test_X, test_y)
    }

    print(f"Best k for dataset {dataset_id}: {best_k} , test acc = {results_tuned_knn['test_acc']:.3f}  (Cross-validation acc={best_k_score:.3f})")

    # save to PDF
    folder_mc = f"dataset_{dataset_id}_method_comparison"
    os.makedirs(folder_mc, exist_ok=True)
    boundary_path = os.path.join(
        folder_mc, f"mc_dataset{dataset_id}_k={best_knn.n_neighbors}.pdf"
    )
    plot_boundary(boundary_path[:-4], best_knn, test_X, test_y,
                  title=f"Train acc= {results_tuned_knn['train_acc']:.3f}, Test acc={results_tuned_knn['test_acc']:.3f}, k={results_tuned_knn['k']}")


    # tuning max_depth
    best_depth, best_depth_score = tune_decision_tree(train_X, train_y, depths)
    best_tree = DecisionTreeClassifier(max_depth=best_depth)
    best_tree.fit(train_X, train_y)

    results_tuned_depth = {
        "depth": best_tree.max_depth,
        "train_acc": best_tree.score(train_X, train_y),
        "test_acc": best_tree.score(test_X, test_y)
    }

    print(f"Best depth for dataset {dataset_id}: {best_depth}, test acc = {results_tuned_depth['test_acc']:.3f}  (Cross-validation acc={best_depth_score:.3f})")
    # save to PDF
    os.makedirs(folder_mc, exist_ok=True)
    boundary_path = os.path.join(
        folder_mc, f"mc_dataset{dataset_id}_depth={best_tree.max_depth}.pdf"
    )
    plot_boundary(boundary_path[:-4], best_tree, test_X, test_y,
                  title=f"Train acc= {results_tuned_depth['train_acc']:.3f}, Test acc={results_tuned_depth['test_acc']:.3f}, depth={results_tuned_depth['depth']}")

    # QDA
    model_qda = QuadraticDiscriminantAnalysis()
    model_qda.fit(train_X, train_y, lda=False)
    preds_qda = model_qda.predict(test_X)
    acc_qda = np.mean(preds_qda == test_y)
    os.makedirs(folder_mc, exist_ok=True)
    boundary_path = os.path.join(
        folder_mc, f"mc_dataset{dataset_id}_QDA.pdf"
    )
    plot_boundary(boundary_path[:-4], model_qda, test_X, test_y, title=f"QDA Test acc={acc_qda:.3f}")
    print(f"QDA Test acc={acc_qda:.3f}")

    # LDA
    model_lda = QuadraticDiscriminantAnalysis()
    model_lda.fit(train_X, train_y, lda=True)
    preds_lda = model_lda.predict(test_X)
    acc_lda = np.mean(preds_lda == test_y)
    os.makedirs(folder_mc, exist_ok=True)
    boundary_path = os.path.join(
        folder_mc, f"mc_dataset{dataset_id}_LDA.pdf"
    )
    plot_boundary(boundary_path[:-4], model_lda, test_X, test_y, title=f"LDA Test acc={acc_lda:.3f}")
    print(f"LDA Test acc={acc_lda:.3f}")


    return results_tuned_knn, results_tuned_depth, acc_qda, acc_lda


def process_breast_cancer_dataset(dataset_id,num_of_neighbours, depths):
    print(f"\n===BREAST CANCER DATASET {dataset_id}===")

    data_X, data_y = make_dataset_breast_cancer(random_state=dataset_id + 41) # manually change the random state to ensure each dataset is different
    train_X, test_X = data_X[:427], data_X[427:] # 569 * 0.75 = 427
    train_y, test_y = data_y[:427], data_y[427:]
    train_X = np.array(train_X)
    train_y = np.array(train_y)

    # tune k
    best_k, best_k_score = tune_knn(train_X, train_y, num_of_neighbours)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(train_X, train_y)

    results_knn_bc={
        "k": best_knn.n_neighbors,
        "train_acc": best_knn.score(train_X, train_y),
        "test_acc": best_knn.score(test_X, test_y)
    }

    print(f"Best k for dataset {dataset_id}: {best_k} , test acc = {results_knn_bc['test_acc']:.3f}  (Cross-validation acc={best_k_score:.3f})")

    # tune max_depth
    best_depth, best_depth_score = tune_decision_tree(train_X, train_y, depths)
    best_tree = DecisionTreeClassifier(max_depth=best_depth)
    best_tree.fit(train_X, train_y)

    results_dt_bc={
        "depth": best_tree.max_depth,
        "train_acc": best_tree.score(train_X, train_y),
        "test_acc": best_tree.score(test_X, test_y)
    }

    print(f"Best depth for dataset {dataset_id}: {best_depth}, test acc = {results_dt_bc['test_acc']:.3f}  (Cross-validation acc={best_depth_score:.3f})")

    # QDA
    model_qda = QuadraticDiscriminantAnalysis()
    model_qda.fit(train_X, train_y, lda=False)
    preds_qda = model_qda.predict(test_X)
    acc_qda = np.mean(preds_qda == test_y)
    print(f"QDA test acc={acc_qda:.3f}")

    # LDA
    model_lda = QuadraticDiscriminantAnalysis()
    model_lda.fit(train_X, train_y, lda=True)
    preds_lda = model_lda.predict(test_X)
    acc_lda = np.mean(preds_lda == test_y)
    print(f"LDA test acc={acc_lda:.3f}")

    return results_knn_bc, results_dt_bc, acc_qda, acc_lda


if __name__ == "__main__":

    num_of_neighbours = [1, 5, 25, 125, 500, 899]
    depths = [1, 2, 4, 6, None]
    tuned_knn_accs = []
    tuned_dt_accs = []
    qda_accs = []
    lda_accs = []

    for dataset_id in range(1,6):# 5 generations of the synthetic dataset
        results_knn, results_dt, qda_accuracy, lda_accuracy = process_dataset(dataset_id, num_of_neighbours,depths)
        tuned_knn_accs.append(results_knn["test_acc"])
        tuned_dt_accs.append(results_dt["test_acc"])
        qda_accs.append(qda_accuracy)
        lda_accs.append(lda_accuracy)

    mean_knn_acc = np.mean(tuned_knn_accs)
    std_knn_acc = np.std(tuned_knn_accs)

    mean_dt_acc = np.mean(tuned_dt_accs)
    std_dt_acc = np.std(tuned_dt_accs)

    mean_qda_acc = np.mean(qda_accs)
    std_qda_acc = np.std(qda_accs)

    mean_lda_acc = np.mean(lda_accs)
    std_lda_acc = np.std(qda_accs)

    print("\n===AVERAGE RESULTS FOR EACH METHOD ± STD (mean over 5 synthetic dataset generations)===")
    print(f"kNN (tuned):   {mean_knn_acc:.3f} ± {std_knn_acc:.3f}")
    print(f"DT (tuned) :   {mean_dt_acc:.3f} ± {std_dt_acc:.3f}")
    print(f"QDA        :   {mean_qda_acc:.3f} ± {std_qda_acc:.3f}")
    print(f"LDA        :   {mean_lda_acc:.3f} ± {std_lda_acc:.3f}")

    # PLOT FOR METHOD COMPARISON
    methods = ["kNN", "Decision Tree", "QDA", "LDA"]
    datasets = range(1, 6)
    accs = [tuned_knn_accs, tuned_dt_accs, qda_accs, lda_accs]
    means = [mean_knn_acc, mean_dt_acc, mean_qda_acc, mean_lda_acc]
    stds = [std_knn_acc, std_dt_acc, std_qda_acc, std_lda_acc]

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    plt.figure(figsize=(10, 6))

    for i, dataset_id in enumerate(datasets):
        plt.scatter(
            methods,
            [accs[j][i] for j in range(len(methods))],
            color=colors[i],
            label=f"Dataset {dataset_id}",
            alpha=0.7,
            s=60
        )

    plt.errorbar(
        methods, means, yerr=stds,
        fmt='o', color='black', capsize=5, markersize=10, label='Mean ± std'
    )

    plt.title("Accuracy comparison for different methods")
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 0.975)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("methods_comparison.pdf", dpi=300)
    print("✅ Plot saved as methods_comparison.pdf")



    print("\n\n=== BREAST CANCER DATASET ===")

    all_results_bc_dt = []
    cv_best_depths_bc_dt = []
    results_knn_bc = []
    results_dt_bc = []
    results_qda_bc = []
    results_lda_bc = []

    for dataset_id in range(1, 6): # 5 generations of the breast cancer dataset
        knn_acc_bc, dt_acc_bc, qda_acc_bc, lda_acc_bc = process_breast_cancer_dataset(dataset_id, num_of_neighbours, depths)
        results_knn_bc.append(knn_acc_bc)
        results_dt_bc.append(dt_acc_bc)
        results_qda_bc.append(qda_acc_bc)
        results_lda_bc.append(lda_acc_bc)

    knn_mean_bc = np.mean([r["test_acc"] for r in results_knn_bc])
    knn_std_bc = np.std([r["test_acc"] for r in results_knn_bc])

    dt_mean_bc = np.mean([r["test_acc"] for r in results_dt_bc])
    dt_std_bc = np.std([r["test_acc"] for r in results_dt_bc])

    qda_mean_bc = np.mean(results_qda_bc)
    qda_std_bc = np.std(results_qda_bc)

    lda_mean_bc = np.mean(results_lda_bc)
    lda_std_bc = np.std(results_lda_bc)

    print("\n===AVERAGE RESULTS FOR EACH METHOD ± STD (mean over 5 breast cancer dataset generations)===")
    print(f"kNN (tuned):   {knn_mean_bc:.3f} ± {knn_std_bc:.3f}")
    print(f"DT (tuned) :   {dt_mean_bc:.3f} ± {dt_std_bc:.3f}")
    print(f"QDA        :   {qda_mean_bc:.3f} ± {qda_std_bc:.3f}")
    print(f"LDA        :   {lda_mean_bc:.3f} ± {lda_std_bc:.3f}")

    # PLOT FOR METHOD COMPARISON
    methods = ["kNN", "Decision Tree", "QDA", "LDA"]
    datasets = range(1, 6)
    accs = [
        [r["test_acc"] for r in results_knn_bc],
        [r["test_acc"] for r in results_dt_bc],
        results_qda_bc,
        results_lda_bc
    ]
    means = [knn_mean_bc, dt_mean_bc, qda_mean_bc, lda_mean_bc]
    stds = [knn_std_bc, dt_std_bc, qda_std_bc, lda_std_bc]

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    plt.figure(figsize=(10, 6))

    for i, dataset_id in enumerate(datasets):
        plt.scatter(
            methods,
            [accs[j][i] for j in range(len(methods))],
            color=colors[i],
            label=f"Dataset {dataset_id}",
            alpha=0.7,
            s=60
        )

    plt.errorbar(
        methods, means, yerr=stds,
        fmt='o', color='black', capsize=5, markersize=10, label='Mean ± std'
    )

    plt.title("Accuracy comparison for different methods on breast cancer dataset")
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("methods_comparison_breast_cancer.pdf", dpi=300)
    print("✅ Plot saved as methods_comparison_breast_cancer.pdf")