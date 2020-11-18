from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from csv import DictReader
import numpy as np
import pandas as pd

fig = plt.figure()
ax = Axes3D(fig)


def fill_cluster(first, second, third):
    varieties = list()
    varieties2 = list()
    if third == '':
        with open('users_info.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            try:
                for row in csv_dict_reader:
                    if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                        varieties.append([int(row[first]), int(row[second])])
            except:
                pass
        get_linear_regression(varieties)
    else:
        with open('users_info.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            try:
                for row in csv_dict_reader:
                    if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                        varieties.append([int(row[first]), int(row[second])])
                        varieties2.append(int(row[third]))
            except:
                pass
        get_linear_3d_regression(varieties,varieties2)


def get_linear_regression(varieties):
    regression_data = np.array(varieties)
    inp = regression_data[:, 0].reshape(-1, 1)
    out = regression_data[:, 1]
    model = LinearRegression().fit(inp, out)
    y_pred = model.predict(inp)
    plt.scatter(inp, out, color='red')
    plt.plot(inp, model.predict(inp), color='grey')
    plt.scatter(inp, y_pred, color='blue')
    plt.show()


def get_linear_3d_regression(varieties,varieties2):
    regression_data = np.array(varieties)
    df2 = pd.DataFrame(regression_data, columns=['Photos', 'Age'])
    df2['Profile_entries'] = pd.Series(varieties2)
    model = LinearRegression().fit(regression_data, varieties2)
    x_surf, y_surf = np.meshgrid(np.linspace(df2.Photos.min(), df2.Photos.max(), 100),
                                 np.linspace(df2.Age.min(), df2.Age.max(), 100))
    only_x = pd.DataFrame({'Photos': x_surf.ravel(), 'Age': y_surf.ravel()})
    fitted_y = model.predict(only_x)
    fitted_y = np.array(fitted_y)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df2['Photos'], df2['Age'], df2['Profile_entries'], c='red', marker='o', alpha=0.5)
    ax.plot_surface(x_surf, y_surf, fitted_y.reshape(x_surf.shape), color='b', alpha=0.3)
    ax.set_xlabel('Photos')
    ax.set_ylabel('Age')
    ax.set_zlabel('Profile_entries')
    plt.show()


def main():
    fill_cluster('Photos', 'Age', 'Profile_entries')


if __name__ == "__main__":
    main()
