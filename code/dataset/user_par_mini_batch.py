"""
User-parallel mini batches dataset implementation
"""

import tensorflow as tf

def get_dataset(batch_size):
    return tf.data.Dataset()

def main():
    dataset = get_dataset(10)
    for datapoint in dataset:
        print(datapoint)
        break

if __name__ == "__main__":
    main()
