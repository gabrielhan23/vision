import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# (x_train, x_val), (y_train, y_val) = mnist.load_data('/Users/gabrielhan/Coding/vision/mnist.npz')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])
model.compile(
    
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=tf.keras.metrics.CategoricalAccuracy()
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
np.set_printoptions(threshold=np.inf)
print(model.summary)
f = open("tensorweights.txt", "w")
f.write(str(np.array(model.trainable_variables,dtype=object)))
f.close()