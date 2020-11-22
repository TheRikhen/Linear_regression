from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from csv import DictReader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def fill_cluster(args, third_arg):
    data = pd.read_csv('users_info.csv')
    new_data = data.dropna(subset=['Age', 'Photos', 'City_id'])
    get_linear_regression(new_data, args)
    get_linear_3d_regression(new_data, args, third_arg)


def get_linear_regression(varieties, args):
    df = pd.DataFrame(varieties, columns=args)
    df['Age'] = df['Age'].astype(int)
    regression_data = np.array(varieties)
    inp = regression_data[:, 0].reshape(-1, 1)
    out = regression_data[:, 1]
    model = LinearRegression().fit(inp, out)
    y_pred = model.predict(inp)
    plt.scatter(inp, out, color='yellow')
    plt.plot(inp, model.predict(inp), color='grey')
    plt.scatter(inp, y_pred, color='red')
    plt.show()


def get_linear_3d_regression(new_data, args, third_arg):
    fig = plt.figure()
    ax = Axes3D(fig)
    df = pd.DataFrame(new_data, columns=args)
    third_arg_massive = pd.DataFrame(new_data, columns=[third_arg])
    third_value = third_arg_massive[third_arg].values.tolist()
    regression_data = np.array(df)
    df[third_arg] = pd.Series(third_value)
    model = LinearRegression().fit(regression_data, third_value)
    x_surf, y_surf = np.meshgrid(np.linspace(df.Photos.min(), df.Photos.max(), 100),
                                 np.linspace(df.Age.min(), df.Age.max(), 100))
    only_x = pd.DataFrame({'Photos': x_surf.ravel(), 'Age': y_surf.ravel()})
    fitted_y = model.predict(only_x)
    fitted_y = np.array(fitted_y)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Photos'], df['Age'], df['Profile_entries'], c='red', marker='o', alpha=0.5)
    ax.plot_surface(x_surf, y_surf, fitted_y.reshape(x_surf.shape), color='b', alpha=0.3)
    ax.set_xlabel('Photos')
    ax.set_ylabel('Age')
    ax.set_zlabel('Profile_entries')
    plt.show()


def main():
    fill_cluster(['Age', 'Photos'], 'Profile_entries')


if __name__ == "__main__":
    main()
