import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compute_fraction_above_threshold(data, threshold):
    """computes fraction of entries > threshold."""
    fraction = np.mean(data > threshold)
    return fraction

def compute_eps_fraction_element(data, eps):
    """Lreturns the eps-fraction-th element."""
    sorted_data = np.sort(data)
    index = int(eps * len(data))
    # print(index)
    # print(len(sorted_data))
    if index == len(sorted_data):
        return sorted_data[-1]
    else:
        return sorted_data[index]
    
def lazy_data_mix(array1, array2, eps1, eps2):
    """This is a lazy way of data mixing without actually creating a whole new dataset."""
    """array_o2 represents less array1 and more array2
       array_o1 represents less array2 and more array1 """
    indices = np.random.permutation(len(array1))
    assert eps1 <= eps2
    # get 1st section from array1 and 2nd section from array2
    cut_position = int(eps2 * len(indices))
    array_o1 = np.concatenate((array1[indices[:cut_position]], array2[indices[cut_position:]] if cut_position < len(indices) else np.array([])))

    cut_position = int(eps1 * len(indices))
    array_o2 = np.concatenate((np.array([]) if cut_position == 0 else array1[indices[:cut_position]], array2[indices[cut_position:]]))

    
    return array_o1, array_o2

def plot_roc_curve(fpr, tpr, auc_score, save_path="roc_curve.png"):
    """Plots and saves the ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model generated cohort", dest="model")
    parser.add_argument("--data", help="Data folder for student cohort", dest="dataset")
    # parser.add_argument("npy_files", nargs="+", help="List of .npy files to process")
    # parser.add_argument("threshold", type=float, help="Threshold value")
    parser.add_argument("--eps1", type=float, help="Closeness parameter", dest='eps1')
    parser.add_argument("--eps2", type=float, help="Farness parameter", dest='eps2')
    parser.add_argument("--expeps", type=float, help="Experiment's Farness parameter", dest='exp_eps')
    parser.add_argument('--verb', dest='verb', help="verbosity")
    parser.add_argument('--seed', dest='seed', type=int, help="seed")

    # args.npy_files = model_files
    args = parser.parse_args()
    eps1 = args.eps1 / 100
    eps2 = args.eps2 / 100
    exp_eps = args.exp_eps / 100
    np.random.seed(args.seed)

    model_files = [f for f in os.listdir(f'./{args.model}') if f.startswith(args.model) and f.endswith('.npy')]
    print(model_files)
    if not model_files:
        print(f"No files found starting with {args.model}")
    
    positive_elems = []
    negative_elems = []
    for model_file in model_files:
        model_path = os.path.join(f'./{args.model}', model_file)
        if os.path.exists(model_path):
            data_file = os.path.join(f'./{args.model}', model_file.replace(args.model, args.dataset))
            if not os.path.exists(data_file):
                if args.verb: print(f"Error: {data_file} not found.")
                continue
            
            # get the files 
            model_data = np.load(model_path)
            data_data = np.load(data_file)
            positive_data, negative_data = lazy_data_mix(model_data, data_data, eps1, eps2)
            epsmedian = compute_eps_fraction_element(positive_data, exp_eps)
            positive_elems.append(epsmedian)
            epsmedian = compute_eps_fraction_element(negative_data, exp_eps)
            negative_elems.append(epsmedian)
           
        else:
            print(f"Error: {model_file} not found.")

    # Assuming positive and negative elements are the true labels and scores
    y_true = [1] * len(positive_elems) + [0] * len(negative_elems)  # 1 for positive, 0 for negative
    y_scores = positive_elems + negative_elems  # Scores from both categories
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print("roc_auc:", roc_auc)
    
    # # Plot ROC curve
    # plot_roc_curve(fpr, tpr, roc_auc, save_path=f"roc_curve_{args.eps}.png")

if __name__ == "__main__":
    main()
