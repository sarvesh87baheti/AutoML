import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate(model, test_ds, y_test, class_names, save_dir):

    loss, acc = model.evaluate(test_ds)
    print("Test Accuracy:", acc)

    preds = model.predict(test_ds)

    if len(class_names) == 2:
        y_pred = (preds > 0.5).astype(int).flatten()
        y_true = y_test
    else:
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    return acc