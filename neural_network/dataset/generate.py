import csv
import numpy as np

NUM_ROWS = 48000
NUM_COLS = 100
WEIGHTS = [(i % 10) for i in range(NUM_COLS)]


def generate_linear_reg_data_and_save_csv(FILENAME='linear_data.csv'):
    with open(FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['Column'+str(i) for i in range(NUM_COLS)]+['WeightedSum']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        data = np.random.uniform(size=(NUM_ROWS, NUM_COLS)).astype(np.float16)
        weights = np.array(WEIGHTS).reshape(-1, 1).astype(np.float16)
        result = data@weights

        for _ in range(NUM_ROWS):
            value = {'Column'+str(i): data[_][i] for i in range(NUM_COLS)}
            value['WeightedSum'] = result[_][0]
            writer.writerow(value)

    print(f"CSV file generated: {FILENAME}")


def sigmoid(x):
    return 1/(1+np.exp(-x))


def generate_logistic_reg_data_and_save_csv(FILENAME='logistic_data.csv'):
    with open(FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['Column'+str(i) for i in range(NUM_COLS)]+['WeightedSum']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        class_0_mean = np.random.randint(1, 5, size=NUM_COLS)
        class_0_cov = np.diag(np.random.uniform(0.5, 1, size=NUM_COLS))
        class_0_data = np.random.multivariate_normal(
            class_0_mean, class_0_cov, NUM_ROWS//2)

        class_1_mean = np.random.randint(6, 10, size=NUM_COLS)
        class_1_cov = np.diag(np.random.uniform(0.5, 1, size=NUM_COLS))
        class_1_data = np.random.multivariate_normal(
            class_1_mean, class_1_cov, NUM_ROWS//2)

        X = np.vstack([class_0_data, class_1_data]).astype(np.float16)
        y = np.hstack(
            [np.zeros(NUM_ROWS//2), np.ones(NUM_ROWS//2)])
        data = np.column_stack((X, y))
        np.random.shuffle(data)
        X = data[:, :-1].astype(np.float16)
        y = data[:, -1].reshape(-1, 1).astype(np.float16)

        writer.writeheader()

        for _ in range(NUM_ROWS):
            value = {'Column'+str(i): X[_][i] for i in range(NUM_COLS)}
            value['WeightedSum'] = y[_][0]
            writer.writerow(value)

    print(f"CSV file generated: {FILENAME}")


if __name__ == "__main__":
    generate_linear_reg_data_and_save_csv(FILENAME='linear_data.csv')
    generate_logistic_reg_data_and_save_csv()
