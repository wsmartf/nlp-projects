import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pandas as pd
import time 
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def read_encode(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                    
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

#### FFNN ####
class FFNN(nn.Module):
    def __init__(self, vocab, words,d_model, d_hidden, dropout ,batch_size, context_size, path):
        super(FFNN,self).__init__() 
    
        self.PATH = path # model path to where the model will be saved
        self.vocab = vocab
        self.words = words
        self.batch_size = batch_size
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(p=dropout)
        self.context_size = context_size

        # model architecture
        self.embeds = nn.Embedding(self.vocab_size,self.d_model)
        self.layer1 = nn.Linear(self.d_model*self.context_size,self.d_model)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(self.d_model, self.vocab_size)
        self.init_weights()

    def forward(self, src):
        x = self.dropout(self.embeds(src).flatten(start_dim=1))
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x
                
    def init_weights(self):
        self.forward(torch.randint(low=0, high=1, size=(1, self.context_size)))

    def train_model(self, loader, optimizer, loss_ce, base_perplexity, mode="train"):
        perplexities= np.zeros((len(loader)))
        for itr, X in enumerate(loader):
            context = X[:,0:3]
            target = X[:,3]
            
            # Clearing the accumulated gradients
            optimizer.zero_grad()

            # get log probabilities over next words
            logits = self.forward(context)
        
            # compute loss function
            loss = loss_ce(logits,target)

            # track perplexity
            p  = torch.exp(loss)
            perplexities[itr] = p

            if mode == "train":
                if p < base_perplexity:
                    base_perplexity = p
                    torch.save(self.state_dict(), self.PATH)
                
                # backward pass and update gradient
                loss.backward()
                optimizer.step()
                    
            if itr % 1000 == 0:
                print("--- Mode: {} | Iteration number: {} | Loss: {} | Perplexity: {} ---".format(mode, itr+1, loss, base_perplexity))
        return perplexities

def generate_blind_prob_dist(data, params, model, threshold):
    bios = []
    bio = []
    for i in range(len(data)):
        bio.append(data[i])
        if data[i] == 121: # <end_bio>
            bios.append(bio)
            bio = []

    labels = []
    for index, bio in enumerate(bios):
        x,y = split_data(bio, params.window+1)
        test_set = np.concatenate([x,y],axis=1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size,shuffle=False)
        bio_prob = test_model_probs(test_loader, model)
        label = '[REAL]' if bio_prob > threshold else '[FAKE]'
        labels.append(label)
    return labels

def generate_prob_dist(data, params, model):
    bios = []
    bio = []
    for i in range(len(data)):
        bio.append(data[i])
        if data[i] == 122 or data[i] == 635: # [FAKE] or [REAL] 
            bios.append(bio)
            bio = []

    real_labels = [] # probability distribution over bios
    probs = []
    for index, bio in enumerate(bios):
        x,y = split_data(bio, params.window+1)
        test_set = np.concatenate([x,y],axis=1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size,shuffle=False)
        bio_prob = test_model_probs(test_loader, model)
        label = '[REAL]' if bio[-1] == 635 else '[FAKE]'
        real_labels.append(label)
        probs.append(bio_prob)
    return real_labels, probs

def test_model_probs(test_loader,model):
    probs = []

    for X in test_loader:
        context = X[:,0:3]
        target = X[:,3]

        with torch.no_grad():

            logits = model(context)

            pred_words = torch.softmax(logits,dim=1) # applying softmax per row
            # get the probability of the target word
            target_word_prob = -1 * torch.log(pred_words[torch.arange(pred_words.shape[0]), target])
            # add prob to list
            probs.extend(target_word_prob)
    return torch.mean(torch.stack(probs)).item()  # torch.stack, stacks each row on top of each other, .mean calculates the mean, .item() gets the value

def train_fnn(model, train, test, params, loss_function, optimizer, epochs):
    model.train()
    epoch_perplexities_train = []
    epoch_perplexities_test = []
    base_perplexity = float('inf')

    # train data loader
    x_train,y_train = split_data(train, params.window+1)
    train_set = np.concatenate([x_train,y_train],axis=1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size,shuffle=True,drop_last=True)
    

    # test data loader
    x_test,y_test = split_data(test, params.window+1)
    test_set = np.concatenate([x_test,y_test],axis=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size,shuffle=False)
    # train the network
    for e in range(epochs):
        epoch_perplexity_train = model.train_model(train_loader, optimizer, loss_function, base_perplexity)
        epoch_perplexity_test = model.train_model(test_loader, optimizer, loss_function, base_perplexity, mode="test")

        epoch_perplexities_train.append(epoch_perplexity_train)
        epoch_perplexities_test.append(epoch_perplexity_test)
        
        torch.save(epoch_perplexities_train, "save/FFNN_epoch_perplexities_train.pt")
        torch.save(epoch_perplexities_test, "save/FFNN_epoch_perplexities_test.pt")

##### LSTM #####
class LSTM(nn.Module):
    def __init__(self, vocab, words,d_model, dropout, d_hidden, seq_len, batch_size, n_layers, savename):
        super().__init__()

        self.PATH = savename + '.pt'
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.embeds = nn.Embedding(self.vocab_size,self.d_model)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_hidden, num_layers=self.n_layers, batch_first=True)
        self.hidden2word = nn.Linear(self.d_hidden, self.vocab_size)
        
        self.embeds.weight = self.hidden2word.weight
        self.init_weights()
        
    def forward(self,src,h,c):
        embeds = self.dropout(self.embeds(src))    
        preds, h = self.lstm(embeds, (h.detach(), c.detach()))
        preds = self.hidden2word(preds)
        
        return [preds,h]
    
    def init_weights(self):
        init_val = 0.2
        self.embeds.weight.data.uniform_(-init_val, init_val)
        self.hidden2word.weight.data.uniform_(-init_val, init_val)
        for i in range(self.n_layers): self.lstm_init(i, init_val)
        print("=====Done with init ====")
    
    def lstm_init(self, i, init_val):
        self.lstm.all_weights[i][1] = torch.FloatTensor(self.d_hidden, self.d_hidden).uniform_(-init_val, init_val) 
        self.lstm.all_weights[i][0] = torch.FloatTensor(self.d_model, self.d_hidden).uniform_(-init_val, init_val) 

    def train_model(self, loader, optimizer, loss_ce, base_perplexity, params, mode="train"):
        perplexities= np.zeros((len(loader)))
        h = torch.zeros(self.n_layers, self.batch_size, self.d_hidden).requires_grad_()
        c = torch.zeros(self.n_layers, self.batch_size, self.d_hidden).requires_grad_()

        for itr, X in enumerate(loader):
            context = X[:,:params.seq_len-1]
            target = X[:,params.seq_len-1]

            # Clearing the accumulated gradients
            optimizer.zero_grad()
            
            # get log probabilities over next words
            logits, (h,c) = self.forward(context, h, c)

            # target = nn.functional.one_hot(target, num_classes=model.vocab_size)
            loss = loss_ce(logits[:,-1,:],target)

            # track perplexity
            p  = torch.exp(loss)
            perplexities[itr] = p

            if mode == "train":
                # backward pass and update gradient
                loss.backward()
                optimizer.step()

                if p < base_perplexity:
                    torch.save(self.state_dict(), self.PATH)
                    base_perplexity = p
            
            if itr % params.printevery == 0:
                print("--- Iteration number: {} | Loss: {} | Perplexity: {} ---".format(itr+1, loss, p))

        return perplexities

def train_lstm(model, train, test, params, loss_function, optimizer, epochs):
    epoch_perplexities_train_lstm = []
    epoch_perplexities_test_lstm = []
    base_perplexity = float('inf')

    train = torch.tensor(train).unfold(dimension=0, size=params.seq_len, step=params.seq_len-1)
    test = torch.tensor(test).unfold(dimension=0, size=params.seq_len, step=params.seq_len-1)


    train_loader = torch.utils.data.DataLoader(train, batch_size=params.batch_size,shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=params.batch_size,shuffle=False, drop_last=True)
    for e in range(epochs):
        print("Epoch: ",e)

        # train
        model.train()
        epoch_perplexity_train = model.train_model(train_loader, optimizer, loss_function, base_perplexity, params)
        epoch_perplexities_train_lstm.append(epoch_perplexity_train)

        # test
        model.eval()
        epoch_perplexity_test = model.train_model(test_loader, optimizer, loss_function, base_perplexity, params, mode="test")
        epoch_perplexities_test_lstm.append(epoch_perplexity_test)


        torch.save(epoch_perplexities_train_lstm, "save/LSTM_epoch_perplexities_train.pt")
        torch.save(epoch_perplexities_test_lstm, "save/LSTM_epoch_perplexities_test.pt")
    
def test_lstm(model, data, params, vocab, words):
    real_wid = words['[REAL]'][0]
    fake_wid = words['[FAKE]'][0]

    data = np.array(data)
    vocab_np = np.array(vocab)
    true_label_inds = [i for i in range(len(data)) if data[i] == real_wid or data[i] == fake_wid]
    
    true_labels = vocab_np[data[true_label_inds]]

    padded_bios = np.pad(data, (0, params.seq_len-1), 'constant')

    processed_data = torch.tensor(padded_bios).unfold(dimension=0, size=params.seq_len-1, step=params.seq_len-1)
    # test_loader = torch.utils.data.DataLoader(data, batch_size=params.batch_size,shuffle=False, drop_last=False)

    h = torch.zeros(model.n_layers, model.d_hidden).requires_grad_()
    c = torch.zeros(model.n_layers, model.d_hidden).requires_grad_()

    output = []
    for itr, X in enumerate(processed_data):
        with torch.no_grad():
            logits, (h,c) = model(X, h, c)
            pred_words_probs = torch.softmax(logits,dim=1) # applying softmax per row
            output.extend(pred_words_probs)

    pred_labels = []

    for i in range(len(true_label_inds)):
        preds = output[true_label_inds[i]]
        real_pred = preds[real_wid]
        fake_pred = preds[fake_wid]
        pred = '[REAL]' if real_pred > fake_pred else '[FAKE]'
        # print(pred, real_pred, fake_pred)
        pred_labels.append(pred)
    
    # print_logit_preds(output, vocab)
    return true_labels, pred_labels

def blind_test_lstm(model, data, params, vocab, words):
    end_bio_wid = words['end_bio'][0]
    real_wid = words['[REAL]'][0]
    fake_wid = words['[FAKE]'][0]

    data = np.array(data)
    true_label_inds = [i+2 for i in range(len(data)) if data[i] == end_bio_wid]

    padded_bios = np.pad(data, (0, params.seq_len-1), 'constant')
    processed_data = torch.tensor(padded_bios).unfold(dimension=0, size=params.seq_len-1, step=params.seq_len-1)

    h = torch.zeros(model.n_layers, model.d_hidden).requires_grad_()
    c = torch.zeros(model.n_layers, model.d_hidden).requires_grad_()

    output = []
    for itr, X in enumerate(processed_data):
        with torch.no_grad():
            logits, (h,c) = model(X, h, c)
            pred_words_probs = torch.softmax(logits,dim=1) # applying softmax per row
            output.extend(pred_words_probs)

    pred_labels = []
    for i in range(len(true_label_inds)):
        preds = output[true_label_inds[i]]
        real_pred = preds[real_wid]
        fake_pred = preds[fake_wid]
        pred = '[REAL]' if real_pred > fake_pred else '[FAKE]'
        # print(pred, real_pred, fake_pred)
        # print(sorted(preds,reverse=True)[:10])
        pred_labels.append(pred)

    # print_logit_preds(output, vocab)

    return pred_labels


#### UTILS ####
#-----------------------#

def split_data(X, window):
  x = []
  y = []
  seq= None
  for i in range(0, len(X)-window):
    seq = X[i:i+window]
    x.append(seq[:-1])
    y.append([seq[-1]])
  return np.array(x),np.array(y)

# Plot a histogram of the data labelled by the class  
def plot_probs(df, threshold):
    dfreal = df[df['label'] == '[REAL]']['prob'].values
    dffake = df[df['label'] == '[FAKE]']['prob'].values

    if len(dfreal) > len(dffake):
        dfreal = dfreal[:len(dffake)]
    else:  
        dffake = dffake[:len(dfreal)]
    X = [dffake, dfreal]

    sns.displot(X, kind='kde', height=5, aspect=1.5, fill=True, legend=False)
    plt.axvline(threshold, color='red', label='Threshold')
    plt.legend(labels=['[REAL]', '[FAKE]', 'Threshold'])
    plt.xlabel('-ve log probability')
    plt.ylabel('Kernel Density Estimate')
    plt.title('Probability Distribution of Bios')
    plt.savefig('save/prob_dist_plot.png')
    plt.show()

def print_logit_preds(output, vocab):
    for i in range(len(output)):
        preds = output[i]
        ind = torch.argmax(preds)
        pred = vocab[ind]
        prob = preds[ind]
        sort_inds = np.argsort(preds)[-5:]
        pred_words = []
        for i in range(len(sort_inds)-1, -1, -1):
            pred_words.append(vocab[sort_inds[i]])
        # print(preds)
        print(ind, pred, prob)
        print("top 5 preds: ", pred_words)
        
def plot_epoch_perplexities(trainLoadFile, testLoadFile, saveFile):
    epoch_perplexities_train = torch.load(trainLoadFile)
    epoch_perplexities_test = torch.load(testLoadFile)

    with torch.no_grad():
        epoch_perplexities_train = np.mean(np.array(epoch_perplexities_train), axis=1)
        epoch_perplexities_test = np.mean(np.array(epoch_perplexities_test), axis=1)

    plt.plot(epoch_perplexities_train, label='train')
    plt.plot(epoch_perplexities_test, label='test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Average Perplexity per Epoch')
    plt.savefig(saveFile)
    plt.show()

def accuracy_score(true_labels, pred_labels):
    correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == true_labels[i]:
            correct += 1
    return correct/len(pred_labels)

def plot_confusion_matrix(true_labels, pred_labels, saveFile=None):
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['[FAKE]', '[REAL]'])
        disp.plot()
        if saveFile is None:
            plt.show()
        else:
            plt.savefig(saveFile)
#-----------------------#
        
def main():
    # use params to decide what model to use, what mode to run in, etc. check main for clarity
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100) #FFNN: 100, LSTM: 64
    parser.add_argument('-d_hidden', type=int, default=200) #FFNN: 200, LSTM: 64
    parser.add_argument('-n_layers', type=int, default=2) #LSTM: 2
    parser.add_argument('-batch_size', type=int, default=200) #FFNN: 200, LSTM: 20
    parser.add_argument('-seq_len', type=int, default=30) #LSTM: 30
    parser.add_argument('-window', type=int, default=3) #FFNN: 3
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.0001) 
    parser.add_argument('-dropout', type=int, default=0.35) 
    parser.add_argument('-model', type=str,default='ffnn') #FFNN: 'ffnn', LSTM: 'lstm'
    parser.add_argument('-mode', type=str,default='eval')

    parser.add_argument('-savename', type=str,default='save/ffnn') #FFNN: 'ffnn', LSTM: 'lstm'
    parser.add_argument('-loadname', type=str,default='models/ffnn') #FFNN: 'ffnn', LSTM: 'lstm'
    parser.add_argument('-trainname', type=str,default='data/mix.train.tok')
    parser.add_argument('-validname', type=str,default='data/mix.valid.tok')
    parser.add_argument('-testname', type=str,default='data/mix.test.tok')
    parser.add_argument('-blindtestname', type=str,default='data/blind.test.tok')

    parser.add_argument('-fnn_threshold', type=float,default=4.5806)
    parser.add_argument('-printevery', type=int, default=1000)

    params = parser.parse_args()    
    torch.manual_seed(0)
    
    [vocab,words,train] = read_encode(params.trainname,[],{},[],3)
    print('vocab: %d train: %d' % (len(vocab),len(train)))
    [vocab,words,test] = read_encode(params.testname,vocab,words,[],-1)
    print('vocab: %d test: %d' % (len(vocab),len(test)))
    [vocab,words,valid] = read_encode(params.validname,vocab,words,[],-1)
    print('vocab: %d valid: %d' % (len(vocab),len(test)))
    params.vocab_size = len(vocab)

    [vocab,words,blind_test] = read_encode(params.blindtestname,vocab,words,[],-1)
    print('vocab: %d blind_test: %d' % (len(vocab),len(test)))

    if params.model == 'ffnn':
        fn = FFNN(vocab, words, params.d_model, params.d_hidden, params.dropout, params.batch_size, params.window, params.savename)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fn.parameters(), lr=params.lr)
        default_threshold = params.fnn_threshold

        if params.mode == 'train':
            # trains & gets epoch perplexities for training set and test set
            train_fnn(fn, train, test, params, loss_function, optimizer, epochs=params.epochs)

            # Plot epoch perplexities for training set and test set
            trainLoadFile = "save/epoch_perplexities_train.pt"
            testLoadFile = "save/epoch_perplexities_test.pt"
            saveFile = "save/FFNN_epoch_perplexities.png"
            plot_epoch_perplexities(trainLoadFile, testLoadFile, saveFile)
            
        elif params.mode == 'eval':
            # Generate predictions for test set
            fn.load_state_dict(torch.load(params.loadname + ".pt"))
            fn.eval()
            labels, probs = generate_prob_dist(test, params, fn)
            pred_labels = ['[REAL]' if x > default_threshold else '[FAKE]' for x in probs]
        
            # Calculate prediction accuracy
            accuracy = accuracy_score(labels, pred_labels)
            print("Accuracy: ", accuracy)

            # Plot confusion matrix  
            plot_confusion_matrix(labels, pred_labels)

        elif params.mode == 'plot': # Get threshold from validation set
            fn.load_state_dict(torch.load(params.loadname + ".pt"))
            fn.eval()
            labels, probs = generate_prob_dist(valid, params, fn)
            plot_probs(pd.DataFrame({'label' : labels, 'prob' : probs}), default_threshold)

        elif params.mode == 'test':
            fn.load_state_dict(torch.load(params.loadname + ".pt"))
            fn.eval()
            labels = generate_blind_prob_dist(blind_test, params, fn, params.fnn_threshold)
            pd.DataFrame(labels).to_csv("save/ffnn_blind_test_labels.csv", index=False, header=False)

    elif params.model == 'lstm':
        params.d_model = 64
        params.d_hidden = 64
        params.batch_size = 20
        params.savename = 'save/lstm'
        params.loadname = 'models/lstm'

        lstm = LSTM(vocab, words, params.d_model, params.dropout, params.d_hidden, params.seq_len, params.batch_size, params.n_layers, params.savename)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm.parameters(), lr=params.lr)

        if params.mode == 'train':
            train_lstm(lstm, train, valid, params, loss_function, optimizer, epochs=params.epochs)

            # Plot epoch perplexities for training set and valid set
            plot_epoch_perplexities("save/LSTM_epoch_perplexities_train.pt", "save/LSTM_epoch_perplexities_test.pt", "save/LSTM_epoch_perplexities.png")
            
        elif params.mode == 'eval':
            lstm.load_state_dict(torch.load(params.loadname + ".pt"))
            lstm.eval()
            true_labels, pred_labels = test_lstm(lstm, test, params, vocab, words)
            # Calculate prediction accuracy
            accuracy = accuracy_score(true_labels, pred_labels)
            print("Accuracy: ", accuracy)

            # Plot confusion matrix
            plot_confusion_matrix(true_labels, pred_labels)

        elif params.mode == 'test':
            lstm.load_state_dict(torch.load(params.loadname + ".pt"))
            lstm.eval()
            pred_labels = blind_test_lstm(lstm, blind_test, params, vocab, words)
            pd.DataFrame(pred_labels).to_csv("save/LSTM_blind_test_labels.csv", index=False, header=False)

if __name__ == "__main__": 
    main()

