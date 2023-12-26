import csv
import numpy as np

NUM_ROWS = 32000
NUM_COLS = 10
WEIGHTS = [i % 10 for i in range(NUM_COLS)]
FILENAME = 'linear_data.csv'


def generate_data_and_save_csv():
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


if __name__ == "__main__":
    generate_data_and_save_csv()
    print(f"CSV file generated: {FILENAME}")
