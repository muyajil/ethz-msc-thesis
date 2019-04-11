from hgru4rec import model_fn
from user_par_mini_batch import input_fn
import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run hgru4rec')

    parser.add_argument('--log_dir', type=str)

    # Dataset specification
    parser.add_argument('--embedding_dict_path', type=str)
    parser.add_argument('--sessions_by_user_prefix', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--min_events_per_session', type=int)

    # Model parameters
    parser.add_argument('--session_rnn_units', type=int)
    parser.add_argument('--user_rnn_units', type=int)
    parser.add_argument('--num_products', type=int)
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--user_rnn_layers', type=int)
    parser.add_argument('--session_rnn_layers', type=int)

    # Model run parameters
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    model_instance = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.log_dir,
        params=vars(args)
    )

    model_instance.train(
        input_fn=lambda: input_fn(
            args.batch_size,
            args.sessions_by_user_prefix,
            args.min_events_per_session,
            embedding_dict_path=args.embedding_dict_path,
            epochs=args.epochs
        ),
        steps=args.train_steps
    )

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Started execution of trainer!')
    main()
    tf.logging.info('Finished execution of trainer!')
