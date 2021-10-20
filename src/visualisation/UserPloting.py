import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.load_dataset import get_dataset, root
import numpy as np

x, x_nan = get_dataset()
week_x = x.set_index("date")
week_x = week_x.groupby("id").apply(lambda df: df.resample("1W").sum()[["usage"]]).reset_index()
users_id = week_x.id.unique()
sample_users = np.random.choice(users_id, 20)
for user_id in sample_users:
    temp_user = week_x[week_x.id == user_id]
    plt.plot(temp_user.usage)
    plt.savefig(root + f"results/user_figures/{user_id}.jpeg")
    plt.close()
