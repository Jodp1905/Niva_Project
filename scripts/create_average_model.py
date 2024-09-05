
def get_average_from_models(models: list[str], model_folder: str):
    """
    Create an average model from a list of models.
    """
    weights = [model.net.get_weights() for model in models]
    avg_weights = [np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                   for weights_list_tuple in zip(*weights)]
    avg_model = initialise_model(input_shape, model_config)
    avg_model.net.set_weights(avg_weights)
    avg_model_path = f'{model_folder}/avg_{model_name}'
    os.makedirs(avg_model_path, exist_ok=True)
    checkpoints_path = os.path.join(
        avg_model_path, 'checkpoints', 'model.ckpt')
    LOGGER.info(f'Average model saved to {avg_model_path}')
    avg_model.net.save_weights(checkpoints_path)

    # Evaluate average model on all folds
    for fold_entry in fold_configurations:
        testing_id = fold_entry['testing_fold']
        LOGGER.info(
            f'Evaluating average model on left-out fold {testing_id}')
        avg_evaluation = avg_model.net.evaluate(ds_folds[testing_id])
        avg_evaluation_dict = dict(
            zip(avg_model.net.metrics_names, avg_evaluation))
        evaluation_path = os.path.join(
            avg_model_path, 'evaluation_avg.json')
        with open(evaluation_path, 'w') as jfile:
            json.dump(avg_evaluation_dict, jfile, indent=4)
            LOGGER.info(
                f'\tEvaluation results for average model saved to {evaluation_path}')
