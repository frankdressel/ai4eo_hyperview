import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import optax
import scipy
from flax.training import checkpoints
from flax.training import train_state

from prediction.ResNetModel import ResNet50
from preprocessing.pipeline import Pipeline, Batch, get_data_pipeline, train_test_split, sample_count


class ResNetExperiment:
    class TrainState(train_state.TrainState):
        batch_stats: Any

    def __init__(self, pipeline: Pipeline, debug=False):
        self.pipeline = pipeline
        self.workdir = "workdir"  # TODO - Figure out the usualk experiment management

        self.print_every_epoch = 1
        self.safe_every_epoch = 10

        self.input_shape = pipeline.input_shape
        self.debug = debug

        model = ResNet50(num_classes=4)

        init_key = jax.random.PRNGKey(0)
        variables = model.init({"params": init_key}, jnp.ones(self.input_shape))
        params = variables["params"]
        batch_stats = variables["batch_stats"]

        # LR schedule
        steps_per_epoch = 1190 // 32 # estimate tbh
        base_learning_rate = 0.05
        num_epochs = 100
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

        tx = optax.adam(learning_rate=schedule_fn)
        self.state = ResNetExperiment.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
        )
        self.model_loss = lambda pred, label: ((pred - label) ** 2).mean()

        if not debug:
            self.train_step = jax.jit(self.train_step)
            self.metrics = jax.jit(self.metrics)

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

    def eval_step(self, batch: Batch):
        predictions = self.state.apply_fn(
            {'params': self.state.params, 'batch_stats': self.state.batch_stats},
            batch.image, mutable=False, train=False)
        return self.metrics(predictions, batch.label)

    def metrics(self, pred, label):
        mse = ((pred - label) ** 2).mean()
        score = mse / self.pipeline.baseline_mse
        return {
            "mse": mse,
            "score": score
        }

    def save_checkpoint(self):
        step = int(self.state.step)
        checkpoints.save_checkpoint(self.workdir, self.state, step, keep=3)

    def train_epochs(self, epochs: int):

        for epoch in range(epochs):
            train_losses, train_metrics = [], []
            eval_metrics = []
            epoch_start = time.time()

            for batch in self.pipeline.train_generator():
                state, loss, predictions = self.train_step(self.state, batch)
                metrics = self.metrics(predictions, batch.label)
                train_losses.append(loss)
                train_metrics.append(metrics)
                self.state = state

            for batch in self.pipeline.test_generator():
                eval_metric = self.eval_step(batch)
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
                print(f"train metrics: \n \t {train_summary}")
                print(f"eval metrics: \n \t {eval_summary}")

            if epoch % self.safe_every_epoch == 0:
                self.save_checkpoint()


if __name__ == '__main__':
    input_shape = (32, 32, 150)


    def preprocess_img(img):
        return jax.image.resize(img, input_shape, "nearest")

    split = train_test_split(sample_count(), 0.3)
    pipeline = get_data_pipeline(split, batch_size=32, preprocess_img=preprocess_img)
    experiment = ResNetExperiment(pipeline, False)
    experiment.train_epochs(100)