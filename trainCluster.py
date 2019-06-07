import tensorflow as tf
import keras

# Define input flags to identify the job and task
from keras.models import Sequential
from keras.layers import Dense, np
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import load_data

tf.app.flags.DEFINE_string("job_name", "worker", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.logging.set_verbosity("DEBUG")
FLAGS = tf.app.flags.FLAGS

cluster = tf.train.ClusterSpec({"ps": ["10.0.0.4:8080"],
                                "worker": [	"10.0.0.6:8080"]})

# Start the server
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

# Configurations
batch_size = 100
learning_rate = 0.0005
training_iterations = 100
num_classes = 2
log_frequency = 10


# Create Keras model
def define_model(num_classes, epochs):
    # Create the model
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model

# Create the optimizer
# We cannot use model.compile and model.fit
def create_optimizer(model, targets):
    predictions = model.output
    loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(targets, predictions))

    # Only if you have regularizers, not in this example
    total_loss = loss * 1.0  # Copy
    for regularizer_loss in model.losses:
        tf.assign_add(total_loss, regularizer_loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Barrier to compute gradients after updating moving avg of batch norm
    with tf.control_dependencies(model.updates):
        barrier = tf.no_op(name="update_barrier")

    with tf.control_dependencies([barrier]):
        grads = optimizer.compute_gradients(
            total_loss,
            model.trainable_weights)
        grad_updates = optimizer.apply_gradients(grads)

    with tf.control_dependencies([grad_updates]):
        train_op = tf.identity(total_loss, name="train")

    return (train_op, total_loss, predictions)
def pre_process(X):
    # normalize inputs from 0-255 to 0.0-1.0
    X = X.astype('float32')
    X = X / 255.0
    return X


def one_hot_encode(y):
    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y, num_classes

# Train the model (a single step)
def train(train_op, total_loss, global_step, step):
        import time
        start_time = time.time()
        X_train_split=np.array_split(X_train, 10)
        X_current_train=X_train_split[step]
        y_train_split = np.array_split(y_train, 10)
        y_current_train=y_train_split[step]
        # print(y_current_train)

        # perform the operations we defined earlier on batch
        loss_value, step_value = sess.run(
            [train_op, global_step],
            feed_dict={
                model.inputs[0]: X_current_train,
                targets: y_current_train})

        if step % log_frequency == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            accuracy = sess.run(total_loss,
                                feed_dict={
                                    model.inputs[0]: X_test,
                                    targets: y_test})
            print("Step: %d," % (step_value + 1),
                  " Iteration: %2d," % step,
                  " Cost: %.4f," % loss_value,
                  " Accuracy: %.4f" % accuracy,
                  " AvgTime: %3.2fms" % float(elapsed_time * 1000 / log_frequency))


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    X, y = load_data.load_datasets()

    # pre process
    X = pre_process(X)

    # one hot encode
    y, num_classes = one_hot_encode(y)

    # split dataset
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    with tf.device("/job:worker/task:0"):
        keras.backend.set_learning_phase(1)
        keras.backend.manual_variable_initialization(True)
        model = define_model(2,12)
        targets = tf.placeholder(tf.float32, shape=[None, 2], name="y-input")
        train_op, total_loss, predictions = create_optimizer(model, targets)
        global_step = tf.contrib.framework.get_or_create_global_step()
        init_op = tf.global_variables_initializer()

    # sv = tf.train.Supervisor( is_chief=(FLAGS.task_index == 0),
    #                          global_step=global_step,
    #                          logdir="/tmp/train_logs",
    #                          save_model_secs=600,
    #                          init_op=init_op)

    print("Waiting for other servers")
    hooks = [tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    init_op = tf.initialize_all_variables()
    with tf.Session("grpc://localhost:8080") as sess:
        # keras.backend.set_session(sess)
        step = 0
        sess.run(init_op)
        train(train_op, total_loss, global_step, step)


    print("done")