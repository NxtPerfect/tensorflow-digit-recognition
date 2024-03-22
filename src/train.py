import tensorflow as tf
import src.model as model
import src.dataset as dataset

def train():
    predictions = model.model(dataset.x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(dataset.y_train[:1], predictions).numpy()

    model.model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.model.fit(dataset.x_train, dataset.y_train, epochs=5)
