"""
   Introduction to Deep Learning
   University of Helsinki

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</path/to/below/modules>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 1000 
LEARNING_RATE = 1e-4 
BATCH_SIZE = 100
REPORT_EVERY = 1
IS_VERBOSE = True


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)


# Generate bag-of-words
# Returns 1 × |𝑉 | PyTorch tensors
def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])



#--- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()
        
        # Two hidden layers with Tanh() as the activation function.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(vocab_size, n_classes), # Params: Inputs, Outputs, Bias
            nn.Tanh(),
            nn.Linear(n_classes, n_classes),
            nn.Tanh(),
            nn.Linear(n_classes, n_classes),
        )
        pass
    
    # Output layer has softmax activation function:
    def forward(self, x):
        x = F.log_softmax(self.linear_relu_stack(x), -1)
        return x



#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)



#--- set up the model ---

model = FFNN(vocab_size, N_CLASSES) # Create a network model
loss_function = nn.NLLLoss() # Loss function
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # Stochastic Gradient Descent as optimizer



#--- Neural Network training ---
for epoch in range(N_EPOCHS):
    total_loss = 0 # Initialize total accumulative loss
    random.shuffle(data['training']) # Shuffle dataset    

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE] # = list of dictionaries
        
        # Zero gradient
        optimizer.zero_grad() 
        
        # Forward
        fx = model.forward(minibatch[i]['BOW']) # Tensor of outputs
        y = label_to_idx(minibatch[i]['SENTIMENT']) # = gold_class
        loss = loss_function(fx, y)
        loss.backward() # Calculate gradients w.r.t. parameters
        optimizer.step() # Update parameters
        total_loss += loss.item() # Add loss for this batch to running total
        
        
        pass
                              
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))



#--- test ---
correct = 0 # Initialize number of correct predictions
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])

        predicted = -1 # Change value to 0, 1 or 2 later
        
        out = model(tweet['BOW']) # = Log Probabilities
        predicted = out.argmax() # Most probable class label ID
        
        if predicted == gold_class:
            correct += 1
        
        if IS_VERBOSE:
            print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))

