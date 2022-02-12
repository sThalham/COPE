import tensorflow.keras as keras
import tensorflow as tf


class CustomModel(tf.keras.Model):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss

    @tf.function
    def train_step(self, data):

        x, y = data

        loss_names = []
        losses = []
        loss_sum = 0
        pose_mask = None

        with tf.GradientTape() as tape:
            predicts = self.model(x)
            for ldx, loss_func in enumerate(self.loss):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y[ldx], dtype=tf.float32)
                if loss_func == 'points':
                    loss, pose_mask = self.loss[loss_func](y_now, predicts[ldx], pose_mask)
                elif loss_func == 'rotations':
                    loss, _ = self.loss[loss_func](y_now, predicts[ldx], pose_mask)
                else:
                    loss = self.loss[loss_func](y_now, predicts[ldx])

                losses.append(loss)
                loss_sum += loss

        grads = tape.gradient(loss_sum, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return_losses = {}
        return_losses["loss"] = loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses[name] = loss

        return return_losses


    def call(self, inputs, training=False):
        x = self.pyrapose(inputs[0])
        if training:
            x = self.pyrapose(inputs[0])
        return x