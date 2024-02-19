import numpy as np
import torch
from models.classifier import  Model_classifier
from models.data_processing.utils import prepare_dataset


device = 'cuda:0'
model_classifier = Model_classifier(device, './output/wae/model_epoch_493.pt').to(device)
state = torch.load('./output/classifier/classifier_50.tar', map_location=torch.device(device))
model_classifier.load_state_dict(state)

def fit_fun(input_file, output_file):

    with open(input_file, 'r') as f:
        seq = f.read().splitlines()

    result = []
    for i in range(len(seq)):
        one = [seq[i]]
        data = prepare_dataset(np.array(one), 105)
        inputs = torch.from_numpy(data)
        with torch.no_grad():
            model_classifier.eval()
            output = model_classifier(inputs.to(device))
            output = output.cpu().detach().numpy().reshape(-1).tolist()[1]
        result.append([str(output), one[0]])

    result = np.array(result)

    with open(output_file, 'w') as f:
        f.write("Probability_score,Sequence\n")
        for row in result:
            f.write(f"{row[0]},{row[1]}\n")

input_file = './prediction/example_sequence.txt'
output_file = './prediction/example_probability.txt'
fit_fun(input_file, output_file)



'''
def fit_fun():
    seq = ['INYIVPHEKDSIAETRQ']
    result = []
    for i in range(len(seq)):
        print(i)
        one = [seq[i]]
        print(one)
        data = prepare_dataset(np.array(one), 75)
        inputs = torch.from_numpy(data)
        with torch.no_grad():
            model_classifier.eval()
            output = model_classifier(inputs.to(device))
            output = output.cpu().detach().numpy().reshape(-1).tolist()[1]
            # (output)
        result.append([str(output), one[0]])
    result = np.array(result)
    return result

X = fit_fun()
print(X)
'''
