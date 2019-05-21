from session_recommendation.main import app, initialize_app
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--embedding_dict_path', type=str)
    parser.add_argument('--params_path', type=str)

    app.run()

    args = parser.parse_args()
    initialize_app(args.model_name, args.model_path, args.embedding_dict_path, args.params_path)