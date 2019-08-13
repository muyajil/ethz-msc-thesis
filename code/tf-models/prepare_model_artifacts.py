from dg_ml_core.file import copy_file, get_paths_with_prefix

DATASET_NAME = 'maxi_dataset'

configs = [
    {
        'variant': 'one_hot',
        'type': 'only_session',
        'model_size': 250,
        'rnn_layers': 1,
        'loss_function': 'top_1',
        'optimizer': 'adam',
        'batch_size': 50,
        'learning_rate': 0.001,
        'early_stop': 20000,
        'gradient_clip': 0.01,
        'min_steps': 10000
    },
    {
        'variant': 'one_hot',
        'type': 'with_user',
        'model_size': 250,
        'rnn_layers': 1,
        'loss_function': 'top_1',
        'optimizer': 'adam',
        'batch_size': 50,
        'learning_rate': 0.001,
        'early_stop': 40000,
        'gradient_clip': 0.01,
        'min_steps': 10000
    },
    {
        'variant': 'embedding',
        'type': 'only_session',
        'model_size': 100,
        'rnn_layers': 1,
        'loss_function': 'top_1',
        'optimizer': 'adam',
        'batch_size': 50,
        'learning_rate': 0.001,
        'early_stop': 20000,
        'gradient_clip': 0.01,
        'min_steps': 50000
    },
    {
        'variant': 'embedding',
        'type': 'with_user',
        'model_size': 100,
        'rnn_layers': 1,
        'loss_function': 'top_1',
        'optimizer': 'adam',
        'batch_size': 50,
        'learning_rate': 0.001,
        'early_stop': 20000,
        'gradient_clip': 0.01,
        'min_steps': 50000
    }
]

for config in configs:
    source_path_base = 'gs://ma-muy/05_logs/'
    source_path_base += config['variant'] + '/'
    source_path_base += config['type'] + '/'
    source_path_base += str(config['model_size']) + '/'
    source_path_base += str(config['rnn_layers']) + '/'
    source_path_base += DATASET_NAME + '/'
    source_path_base += config['loss_function'] + '/'
    source_path_base += config['optimizer'] + '/'
    source_path_base += 'bs_' + str(config['batch_size']) + '/'
    source_path_base += 'lr_' + str(config['learning_rate']) + '/'
    source_path_base += 'es_' + str(config['early_stop']) + '/'
    source_path_base += 'gc_' + str(config['gradient_clip']) + '/'
    source_path_base += 'ms_' + str(config['min_steps']) + '/'
    source_path_base += 'exported_model/'

    possible_models = get_paths_with_prefix(source_path_base)
    possible_models = [x.replace(source_path_base, '') for x in possible_models]
    possible_models = [int(x.split('/')[0]) for x in possible_models]
    latest_model = max(possible_models)
    
    source_path_base += str(latest_model) + '/'

    target_path_base = 'gs://ma-muy/04_model_artifacts/'
    if config['variant'] == 'embedding':
        target_path_base += config['type'] + '_embedding/1/'
    else:
        target_path_base += config['type'] + '/1/'
    
    for source_path in get_paths_with_prefix(source_path_base):
        target_path = source_path.replace(source_path_base, target_path_base)
        print('Copying {} to {}'.format(source_path, target_path))
        copy_file(source_path, target_path)
