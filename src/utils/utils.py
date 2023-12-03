import json
import os
import datetime
import numpy as np
import plotly.graph_objects as go

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def moving_average(x, n=100):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def save_training(experiment_name, config, table, rewards, n=100, show_plot=False):
    fig = make_learning_plot(experiment_name, rewards, n=n, show_plot=show_plot)
    suffix = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
    experiment_path = os.path.join(PROJECT_DIR, "res", experiment_name, suffix)
    os.makedirs(experiment_path, exist_ok=True)

    if isinstance(table, np.matrix):
        path = os.path.join(experiment_path, f"qtable.npy")
        with open(path, "wb") as f:
            np.save(f, table)
    elif isinstance(table, dict):
        for k, v in table.items():
            path = os.path.join(experiment_path, f"{k}.npy")
            with open(path, "wb") as f:
                np.save(f, v)

    path = os.path.join(experiment_path, f"plot.html")
    fig.write_html(path)

    path = os.path.join(experiment_path, f"plot.png")
    fig.write_image(path)

    path = os.path.join(experiment_path, f"config.json")
    with open(path, "w") as f:
        json.dump(config, f, default=str, indent=4)


def make_learning_plot(experiment, rewards, n=1000, show_plot=False):
    # scatter = go.Scatter(
    #     x=np.arange(len(rewards)),
    #     y=rewards,
    #     mode='lines',
    #     name='episodes',
    #     marker=dict(color='#a1d9f7')
    # )

    moving_avg = go.Scatter(
        x=np.arange(n, len(rewards)),
        y=moving_average(rewards),
        mode='lines',
        name=f'moving average of last {n} episodes'
    )

    layout = go.Layout(
        xaxis=dict(title='Episode'),
        yaxis=dict(title='Reward'),
        legend=dict(x=0, y=1),
        title=f'{experiment}'
    )

    fig = go.Figure(data=[moving_avg], layout=layout)

    if show_plot:
        fig.show()

    return fig

