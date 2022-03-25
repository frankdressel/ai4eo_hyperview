import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training import checkpoints
from flax.training import train_state
from skimage.transform import resize

from prediction.ResNetModel import ResNet50
from preprocessing.pipeline import Pipeline, Batch, get_data_pipeline, train_test_split, sample_count, get_test_data, \
    get_train_data


class ResNetExperiment:
    class TrainState(train_state.TrainState):
        batch_stats: Any
        augment_key: jax.random.PRNGKey

    def __init__(self, pipeline: Pipeline, debug=False):
        self.pipeline = pipeline
        self.workdir = "workdir"  # TODO - Figure out the usual experiment management

        self.print_every_epoch = 10
        self.safe_every_epoch = 100

        self.input_shape = pipeline.input_shape
        self.debug = debug

        model = ResNet50(num_classes=4)

        init_key = jax.random.PRNGKey(0)
        variables = model.init({"params": init_key}, jnp.ones(self.input_shape))
        params = variables["params"]
        batch_stats = variables["batch_stats"]

        # LR schedule
        steps_per_epoch = 1213 // self.input_shape[0]  # estimate tbh
        base_learning_rate = 0.02
        num_epochs = 500
        warmup_epochs = 10
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=base_learning_rate,
            transition_steps=warmup_epochs * steps_per_epoch)
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * steps_per_epoch])
        self.lr_fun = schedule_fn
        tx = optax.adam(learning_rate=schedule_fn)
        self.state = ResNetExperiment.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
            augment_key=jax.random.PRNGKey(42)
        )
        self.model_loss = lambda pred, label: ((pred - label) ** 2).mean()

        if not debug:
            self.train_step = jax.jit(self.train_step)
            self.eval_step = jax.jit(self.eval_step)

    def train_step(self, state: TrainState, batch: Batch):
        def loss_fn(params):
            pred, new_model_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, batch.image, mutable=['batch_stats'])
            loss = self.model_loss(pred, batch.label)

            weight_penalty_params = jax.tree_leaves(params)
            weight_decay = 0.0
            weight_l2 = sum([jnp.sum(x ** 2)
                             for x in weight_penalty_params
                             if x.ndim > 1])
            weight_penalty = weight_decay * 0.5 * weight_l2
            loss = loss + weight_penalty

            return loss, (new_model_state, pred)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grad = grad_fn(state.params)

        loss = aux[0]
        new_model_state, preds = aux[1]
        new_state = state.apply_gradients(grads=grad, batch_stats=new_model_state["batch_stats"])
        return new_state, loss, preds

    def eval_step(self, state: TrainState, batch: Batch):
        predictions = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            batch.image, mutable=False, train=False)
        return self.metrics(predictions, batch.label)

    def metrics(self, pred, label):
        mse = ((pred - label) ** 2).mean(axis=0)
        score = mse / self.pipeline.baseline_mse
        return {
            "mse": mse,
            "score": score
        }

    def _get_random_crop(_, image_shape, key: jax.random.PRNGKey) -> list:
        # crop some random area with a similar aspect ratio
        area = (image_shape[1] * image_shape[0])
        area_key, key = jax.random.split(key)
        target_area = jax.random.uniform(area_key, [], minval=0.33, maxval=1.0) * area
        aspect_ratio_key, key = jax.random.split(key)
        aspect_ratio = jnp.exp(
            jax.random.uniform(aspect_ratio_key, [], minval=jnp.log(3 / 4), maxval=jnp.log(4 / 3)))

        w = (target_area * aspect_ratio).round().astype(jnp.int32)
        h = (target_area / aspect_ratio).round().astype(jnp.int32)

        w = jax.lax.min(w, image_shape[1])
        h = jax.lax.min(h, image_shape[0])

        offset_w_k, offset_h_key = jax.random.split(key)

        offset_w = jax.random.uniform(offset_w_k,
                                      (),
                                      minval=0.,
                                      maxval=(image_shape[1] - w + 1).astype(float),
                                      ).round().astype(jnp.int32)
        offset_h = jax.random.uniform(offset_h_key,
                                      (),
                                      minval=0.,
                                      maxval=(image_shape[0] - h + 1).astype(float),
                                      ).round().astype(jnp.int32)

        return [offset_h, offset_w, h, w]

    def augment(self, image: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        crop_key, flip_key = jax.random.split(key)
        offset_h, offset_w, h, w = self._get_random_crop(image.shape, crop_key)
        image = image[offset_h:offset_h + h, offset_w:offset_w + w]
        resized = jax.image.resize(image, self.input_shape, jax.image.ResizeMethod.LINEAR)
        flipped = jnp.where(jax.random.uniform(flip_key, ()) > 0.5, jnp.fliplr(resized), resized)
        return flipped

    def save_checkpoint(self):
        step = int(self.state.step)
        checkpoints.save_checkpoint(self.workdir, self.state, step, keep=3, overwrite=True)

    def train_epochs(self, epochs: int):

        for epoch in range(epochs):
            train_losses, train_metrics = [], []
            eval_metrics = []
            epoch_start = time.time()

            for batch in self.pipeline.train_generator():
                augment_key, state_key = jax.random.split(self.state.augment_key)
                self.state.replace(augment_key=state_key)
                augmented = self.augment(batch.image, augment_key)
                augmented_batch = Batch(augmented, batch.label)
                state, loss, predictions = self.train_step(self.state, augmented_batch)
                metrics = self.metrics(predictions, batch.label)
                train_losses.append(loss)
                train_metrics.append(metrics)
                self.state = state

            for batch in self.pipeline.test_generator():
                eval_metric = self.eval_step(self.state, batch)
                eval_metrics.append(eval_metric)

            # This doesn't seem like the clearest way to do it.
            train_summary = {
                f"train_{k}": v
                for k, v in jax.tree_multimap(lambda *x: jnp.array(x).mean(), *train_metrics).items()
            }
            eval_summary = {
                f"eval_{k}": v
                for k, v in jax.tree_multimap(lambda *x: jnp.array(x).mean(), *eval_metrics).items()
            }

            if epoch % self.print_every_epoch == 0:
                print(f"Epoch {epoch}, took: {time.time() - epoch_start}")
                print(f"LR: {self.lr_fun(self.state.step)}")
                print(f"train metrics: \n \t {train_summary}")
                print(f"eval metrics: \n \t {eval_summary}")

            if epoch % self.safe_every_epoch == 0:
                self.save_checkpoint()

    def predict(self, data: np.ndarray):
        predictions = self.state.apply_fn(
            {'params': self.state.params, 'batch_stats': self.state.batch_stats},
            data, mutable=False, train=False)
        return predictions


def main():
    input_shape = (64, 64, 150)

    def preprocess_img(img):
        return resize(img.astype(np.float32), input_shape)

    split = train_test_split(sample_count(), 0.3)
    pipeline = get_data_pipeline(split, batch_size=64, preprocess_img=preprocess_img)
    experiment = ResNetExperiment(pipeline, False)
    experiment.train_epochs(500)

    test_data = get_test_data(preprocess_fn=preprocess_img, mean=pipeline.images.mean, var=pipeline.images.std_var)
    test_preds = experiment.predict(test_data)

    # denormalize
    denorm_test_preds = (test_preds * pipeline.labels.std_dev) + pipeline.labels.mean
    pd.DataFrame(denorm_test_preds).to_csv("workdir/test_pred.csv")

    # also pred training data as test
    train_data = get_train_data(preprocess_fn=preprocess_img, mean=pipeline.images.mean, var=pipeline.images.std_var)
    train_preds = experiment.predict(train_data)
    denorm_train_preds = (train_preds * pipeline.labels.std_dev) + pipeline.labels.mean
    pd.DataFrame(denorm_train_preds).to_csv("workdir/train_pred.csv")


if __name__ == '__main__':
    main()
