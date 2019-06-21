from hgru4rec import HGRU4Rec
from user_par_mini_batch import UserParallelMiniBatchDataset
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
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--clip_gradients_at', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--loss_function', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    params = vars(args)
    json.dump(vars(args), open(args.log_dir + 'params.json', 'w'))

    model_instance = HGRU4Rec(params)
    model_instance.logger.info('Started execution of trainer!')
    model_instance.setup_model()

    train_dataset = UserParallelMiniBatchDataset(
        args.batch_size,
        args.train_prefix)

    valid_dataset = UserParallelMiniBatchDataset(
        args.batch_size,
        args.eval_prefix)

    model_instance.train(train_dataset, valid_dataset)

    json.dump(model_instance.user_embeddings, open(args.log_dir + 'user_embeddings.json', 'w'))
    json.dump(model_instance.session_embeddings, open(args.log_dir + 'session_embeddings.json', 'w'))
    model_instance.logger.info('Finished execution of trainer!')

if __name__ == "__main__":
    main()
