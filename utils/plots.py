import matplotlib.pyplot as plt


def plot_metrics(metrics):

    labels = ["accuracy", "hallucination_rate"]
    values = [metrics["accuracy"], metrics["hallucination_rate"]]

    plt.bar(labels, values)

    plt.title("Model Performance")

    filename = "experiments/results/metrics_plot.png"

    plt.savefig(filename)

    plt.show()

    print("Saved metrics plot to", filename)