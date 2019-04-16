from hgru4rec import model_fn
from user_par_mini_batch import input_fn
import tensorflow as tf
import argparse
import os


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

    # Model run parameters
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    model_instance = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.log_dir,
        params=vars(args)
    )

    # model_instance.train(
    #     input_fn=lambda: input_fn(
    #         args.batch_size,
    #         args.train_prefix,
    #         epochs=args.epochs
    #     ),
    #     steps=args.train_steps
    # )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            args.batch_size,
            args.train_prefix,
            epochs=args.epochs
        ))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            args.batch_size,
            args.eval_prefix,
        )
    )

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
