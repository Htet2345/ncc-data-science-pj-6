"""Held-out classification diagnostics rendered to PNG."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve


def chart(title, subtitle):
    figure = Figure(figsize=(10, 6), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    figure.subplots_adjust(left=0.15, right=0.92, bottom=0.23, top=0.78)
    figure.text(0.1, 0.94, title, fontsize=19, weight="bold", color="#15283F")
    figure.text(0.1, 0.88, subtitle, fontsize=10, color="#526275")
    figure.text(0.1, 0.045, "DATA SCIENCE PJ6 | Historical Titanic data - OpenML 40945", fontsize=9, color="#526275")
    return figure, axes


def save_plots(actual, probability, matrix, output):
    figure, axes = chart("Titanic prediction confusion matrix", "Held-out passengers only; 0 = did not survive, 1 = survived")
    axes.imshow(matrix, cmap="Blues", vmin=0)
    axes.set_xticks([0, 1], ["0 - Did not survive", "1 - Survived"])
    axes.set_yticks([0, 1], ["0 - Did not survive", "1 - Survived"])
    axes.set(xlabel="Predicted class", ylabel="Actual class")
    for i in range(2):
        for j in range(2):
            axes.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=22, color="white" if matrix[i, j] > matrix.max() / 2 else "#15283F")
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    figure.clear()

    figure, axes = chart("Titanic survival ROC curve", "Trade-off across thresholds; positive class is survived")
    false_positive, true_positive, _ = roc_curve(actual, probability)
    axes.plot(false_positive, true_positive, color="#2457A7", linewidth=2.3, label="Logistic regression")
    axes.plot([0, 1], [0, 1], "--", color="#738096", label="Chance reference")
    axes.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.03))
    axes.grid(color="#E5EAF0")
    axes.legend(frameon=False, loc="lower right")
    figure.savefig(output / "roc_curve.png", dpi=160)
    figure.clear()
