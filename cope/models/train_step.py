import tensorflow.keras as keras
import tensorflow as tf


class CustomModel(tf.keras.Model):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.loss_tracker = tf.keras.metrics.Mean()
        self.points_tracker = tf.keras.metrics.Mean()
        self.cls_tracker = tf.keras.metrics.Mean()
        self.translations_tracker = tf.keras.metrics.Mean()
        self.rotations_tracker = tf.keras.metrics.Mean()
        self.consistency_tracker = tf.keras.metrics.Mean()
        self.projection_tracker = tf.keras.metrics.Mean()

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
        keypoints_gt = None

        with tf.GradientTape() as tape:
            predicts = self.model(x)
            for ldx, loss_func in enumerate(self.loss):
                loss_names.append(loss_func)
                if loss_func != 'projection':
                    y_now = tf.convert_to_tensor(y[ldx], dtype=tf.float32)
                if loss_func == 'points':
                    keypoints_gt = y_now
                    loss, pose_mask = self.loss[loss_func](y_now, predicts[ldx], pose_mask)
                elif loss_func == 'rotations':
                    loss, _ = self.loss[loss_func](y_now, predicts[ldx], pose_mask)
                elif loss_func == 'projection':
                    loss = self.loss[loss_func](keypoints_gt, predicts[ldx], pose_mask)
                else:
                    loss = self.loss[loss_func](y_now, predicts[ldx])

                losses.append(loss)
                loss_sum += loss

        grads = tape.gradient(loss_sum, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.loss_tracker.update_state(loss_sum)
        self.points_tracker.update_state(losses[0])
        self.cls_tracker.update_state(losses[1])
        self.translations_tracker.update_state(losses[2])
        self.rotations_tracker.update_state(losses[3])
        self.consistency_tracker.update_state(losses[4])
        self.projection_tracker.update_state(losses[5])

        return {"loss": self.loss_tracker.result(), "points": self.points_tracker.result(), "cls": self.cls_tracker.result(), "translations": self.translations_tracker.result(), "rotations": self.rotations_tracker.result(), "consistency": self.consistency_tracker.result(), "projection": self.projection_tracker.result()}

        #return_losses = {}
        #return_losses["loss"] = loss_sum
        #for name, loss in zip(loss_names, losses):
        #    return_losses[name] = loss

        #return return_losses

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.points_tracker, self.cls_tracker, self.translations_tracker, self.rotations_tracker, self.consistency_tracker, self.projection_tracker]

    def call(self, inputs, training=False):
        x = self.pyrapose(inputs[0])
        if training:
            x = self.pyrapose(inputs[0])
        return x