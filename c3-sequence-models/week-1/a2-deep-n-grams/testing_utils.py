from dlai_tools.testing_utils import summary, comparator


def model_summary(model):
    result = summary(model)
    for i in range(len(result)):
        if result[i][0] == 'GRU':
            result[i][3] = f'return_sequences={result[i][3]}'
            result[i].append(f'return_state={model.layers[i].return_state}')
    return result
