from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from csv import DictReader
import numpy as np

varieties = list()


def fill_cluster(first, second):
    global varieties
    varieties.clear()
    with open('users_info.csv', 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            if row['Age'] != '' and row['City'] != '' and row['City_id'] != '':
                varieties.append([int(row[first]), int(row[second])])


def get_linear_regression():
    x, y = zip(*varieties)
    inp = np.array(x).reshape((-1, 1))
    out = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(inp, out, test_size=1 / 3, random_state=0)
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    plt.scatter(x_train, y_train, color='red')
    plt.scatter(inp, out, color='yellow')
    plt.plot(x_train, model.predict(x_train), color='blue')
    plt.scatter(x_test, y_pred, color='green')
    # plt.scatter(inp, out, color='red')
    plt.show()


def main():
    fill_cluster('Photos', 'Age')
    get_linear_regression()


if __name__ == "__main__":
    main()