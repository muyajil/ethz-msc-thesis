from session_recommendation.main import app, initialize_app
import json

if __name__ == "__main__":
    config = json.load(open('/config.json'))

    app.run()

    initialize_app(config['model_name'], config['model_path'], config['embedding_dict_path'], config['params_path'])
