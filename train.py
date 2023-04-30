import sys

import tensorflow as tf
import pandas as pd

from absl import app
from absl import flags

from data import create_dataset
from model import create_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras import backend as K

from plan import load_plan
from lr import create_warm_linear_schedule

flags.DEFINE_string('plan', None, 'toml file')
flags.mark_flags_as_required(['plan'])
FLAGS = flags.FLAGS


def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)


class LogLrCallback(Callback):
  def on_epoch_end(self, epoch, logs):
    tf.summary.scalar('learning_rate', self.model.optimizer.lr, epoch)


def main(argv):
  plan = load_plan(FLAGS.plan)
  tplan = plan.train
  dplan = plan.data

  ds_train = create_dataset(['foo3.rio-0000[0-8]-of-00010'], batch=dplan.batch, shuffle=dplan.batch * 10)
  ds_val = create_dataset(['foo3.rio-00009-of-00010'], batch=dplan.batch, shuffle=None)
  model = create_model(plan.model)
  model.summary()

  callbacks = [TerminateOnNaN(),
               LogLrCallback()
               ]
  callbacks.append(
    TensorBoard(
      log_dir="/tmp/logs",
      histogram_freq=0,
      write_graph=False,
      write_images=False,
      write_steps_per_second=True,
      update_freq=1))


  callbacks.append(create_warm_linear_schedule(tplan))

  model.compile(optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  history = model.fit(x=ds_train,
                      epochs=tplan.epochs,
                      steps_per_epoch=tplan.steps_per_epoch,
                      validation_data=ds_val,
                      validation_steps=tplan.val_steps,
                      callbacks=callbacks)

  df = pd.DataFrame(history.history)
  df_to_csv(df, '/dev/stdout')


if __name__ == "__main__":
  app.run(main)
