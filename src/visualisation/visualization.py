import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def plot_result(temp_df: pd.DataFrame):
    labels = temp_df.name

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    list_reacts = []
    for i, col in enumerate(temp_df.drop(columns=["Method"]).columns):
        model = MinMaxScaler()
        y = model.fit_transform(temp_df[col].to_numpy().reshape(-1, 1)).squeeze()
        list_reacts.append(ax.bar(x + (i * width), y, width, label=col))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('value')
    ax.set_title('measurements')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for react in list_reacts:
        ax.bar_label(react, padding=3)

    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50, 50, 960, 640)
    plt.tight_layout()
    plt.show()
