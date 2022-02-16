import tensorflow as tf
import numpy as np
import fedput


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float64)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=5, )

        ds = ds.map(self.split_window)

        return ds


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


WindowGenerator.train = train
WindowGenerator.test = test

lumos = fedput.collect_data_lumos()
column_indices = {name: i for i, name in enumerate(lumos.columns)}

n = len(lumos)
train_df = lumos[0:int(n * 0.7)]
test_df = lumos[int(n * 0.7):]

num_features = lumos.shape[1]
w2 = WindowGenerator(input_width=5, label_width=1, shift=1, train_df=train_df, test_df=test_df,
                     label_columns=['Throughput'])

tfds = w2.train

X_train = list(map(lambda x: x[0], tfds))
y_train = list(map(lambda x: x[-1], tfds))
y_train = np.array(y_train)


# print('X_train', np.array(X_train))

def elemt_arr(y_train):
    val = []
    for e in y_train:
        for j in e:
            for k in j:
                ele = []
                for l in k:
                    ele.append(l)
                    val.append(ele)
    return np.array(val)


val = elemt_arr(y_train)
print(val)
