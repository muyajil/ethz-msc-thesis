from hgru4rec import model_fn, mrr_metric
from user_par_mini_batch import input_fn
import tensorflow as tf
import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser(description='Run hgru4rec')

    parser.add_argument('--log_dir', type=str)

    # Dataset specification
    parser.add_argument('--train_prefix', type=str)
    parser.add_argument('--eval_prefix', type=str)
    parser.add_argument('--batch_size', type=int)

    # Model parameters
    parser.add_argument('--session_rnn_units', type=int)
    parser.add_argument('--user_rnn_units', type=int)
    parser.add_argument('--num_products', type=int)
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--user_rnn_layers', type=int)
    parser.add_argument('--session_rnn_layers', type=int)
    parser.add_argument('--user_dropout', type=float)
    parser.add_argument('--session_dropout', type=float)
    parser.add_argument('--init_dropout', type=float)
    parser.add_argument(
        '--use_user_rnn',
        default=False,
        action='store_true')

    # Model run parameters
    parser.add_argument('--num_predictions', type=int)
    parser.add_argument('--min_train_steps', type=int)
    parser.add_argument('--eval_every_steps', type=int)
    parser.add_argument('--max_steps_without_increase', type=int)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--clip_gradients_at', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--loss_function', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True))

    trainingConfig = tf.estimator.RunConfig(
        session_config=config,
        save_checkpoints_steps=args.eval_every_steps)

    model_instance = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.log_dir,
        params=vars(args),
        config=trainingConfig)

    json.dump(vars(args), open(args.log_dir + 'params.json', 'w'))

    early_stopping_hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator=model_instance,
        metric_name='eval_metrics/mrr',
        max_steps_without_increase=args.max_steps_without_increase,
        run_every_steps=args.eval_every_steps,
        run_every_secs=None,
        min_steps=args.min_train_steps
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            args.batch_size,
            args.train_prefix,
            epochs=args.epochs),
        max_steps=args.train_steps,
        hooks=[early_stopping_hook])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            args.batch_size,
            args.eval_prefix,
            epochs=1),
        steps=None,
        throttle_secs=0,
        start_delay_secs=0)

    tf.estimator.train_and_evaluate(
        estimator=model_instance,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Started execution of trainer!')
    main()
    tf.logging.info('Finished execution of trainer!')
