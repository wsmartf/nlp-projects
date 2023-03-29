from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class Classify():  
    def __init__(self):
        torch.manual_seed(0)

        # classification training  parameters
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        for param in self.model.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(768,2)

        for param in self.linear.parameters():
            param.requires_grad = True

        self.optimizer = optim.Adam(self.linear.parameters(), lr=3e-5)
        print("Model Initialized")

        self.train_set = self.preprocess_data('train')
        self.test_set = self.preprocess_data('test')
        self.valid_set = self.preprocess_data('valid')
    
    def preprocess_data(self, data_type):
        answers = ['A','B','C','D']

        file_name = 'data/' + data_type + '_complete.jsonl'
        data = []
    
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)
            
            base = result['fact1'] + ' [SEP] ' + result['question']['stem']
            ans = answers.index(result['answerKey'])
            
            obs = []
            for j in range(4):
                text = base + ' ' + result['question']['choices'][j]['text'] + ' [SEP]'
                if j == ans:
                    label = 1
                else:
                    label = 0
                obs.append([text,label])
            data.append(obs)
        return data

    def train(self):
        train_loss = []
        train_accuracy = []
        valid_accuracy = []
        for epoch in range(15):
            print(f"Starting training epoch {epoch}")
            epoch_train_loss, epoch_train_accuracy = self.train_model(self.model, self.linear, self.train_set, self.tokenizer, self.optimizer)
            
            with torch.no_grad():
                print(f"Validating epoch {epoch}")
                epoch_valid_accuracy = self.evaluate_model(self.model, self.linear, self.valid_set, self.tokenizer)
                train_loss.append(np.copy(epoch_train_loss))
                train_accuracy.append(np.copy(epoch_train_accuracy))
                valid_accuracy.append(np.copy(epoch_valid_accuracy))

                if epoch == 0 or (epoch > 0 and valid_accuracy[-1] > valid_accuracy[-2]):
                    print("Saving model...")
                    torch.save(self.linear.state_dict(), 'save/linear.pt')
                pd.DataFrame({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy}).to_csv('save/results.csv', index=False)
            
            print(f'Epoch: {epoch} complete | Train Loss: {train_loss[-1]} | Train Accuracy: {train_accuracy[-1]} | Valid Accuracy: {valid_accuracy[-1]}')

    def test(self, model_path='save/linear.pt'):
        
        ## Load weights and test model
        test_model = BertModel.from_pretrained("bert-base-uncased")
        test_linear = torch.nn.Linear(768,2)
        if model_path is not None: 
            test_linear.load_state_dict(torch.load('models/linear.pt'))
        test_accuracy = self.evaluate_model(test_model, test_linear, self.test_set, self.tokenizer)
        valid_accuracy = self.evaluate_model(test_model, test_linear, self.valid_set, self.tokenizer)
        return valid_accuracy, test_accuracy

    def evaluate_model(self, model, linear, data, tokenizer):
        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for i in range(len(data)):
                obs = data[i]
                text = [x[0] for x in obs]
                labels = torch.tensor([x[1] for x in obs])

                inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
                outputs = model(**inputs)

                last_hidden = outputs.last_hidden_state[:,0,:]
                logits = linear(last_hidden)
    
                probs = logits.softmax(dim=1)
                maxind_pred = torch.argmax(probs, dim=0)[1]
                maxind_true = torch.argmax(labels, dim=0)
                
                if maxind_pred == maxind_true:
                    correct += 1
                total += 1

            return correct / total

    def train_model(self, model, linear, data, tokenizer, optimizer):
        model.train()

        # for calculating avg accuracy every N iterations
        total_epoch_iters = 0
        correct_preds = 0
        interval_correct = 0

        # for calculating avg loss every N iterations
        total_epoch_loss = 0
        interval_loss = 0
        interval_iters = 0

        for i in range(len(data)):
            
            obs = data[i]
            text = [x[0] for x in obs]
            labels = torch.tensor([x[1] for x in obs])

            inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
            
            optimizer.zero_grad()
            outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state[:,0,:]

            logits = linear(last_hidden)

            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_epoch_loss += loss

            loss.backward()
            optimizer.step()
            
            interval_loss += loss.item()
            interval_iters += 1

            probs = logits.softmax(dim=1)
            maxind_pred = torch.argmax(probs, dim=0)[1]
            maxind_true = torch.argmax(labels, dim=0)

            if maxind_pred == maxind_true:
                interval_correct +=1
                correct_preds += 1

            if i % 100 == 0 or i == len(data)-1:
                print(f"Iter: {i} | Loss: {interval_loss/interval_iters} | Accuracy: {interval_correct/interval_iters}")
                interval_iters = 0
                interval_loss = 0
                interval_correct = 0

            total_epoch_iters += 1
        
        metrics = total_epoch_loss/total_epoch_iters, correct_preds/total_epoch_iters
        return metrics


class Generate():
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.loss = torch.nn.CrossEntropyLoss()

        self.label_inds = {'A':32, 'B':33, 'C':34, 'D':35}
        self.labels = ['A', 'B', 'C', 'D']

        self.train_set = self.preprocess_data('train')
        self.valid_set = self.preprocess_data('valid')
        self.test_set = self.preprocess_data('test')

    def preprocess_data(self, data_type):
        file_name = 'data/' + data_type + '_complete.jsonl'
        data = []
    
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)
            
            obs = ''
            base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
            ans = result['answerKey']
            choices = ''
            for j in range(len(result['question']['choices'])):
                choices = choices + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '

            obs = base + choices + '[ANSWER]' + ans
            data.append(obs)
        return data

    def preprocess_data_text_file(self, data_type):
        file_name = 'data/' + data_type + '_complete.jsonl'
        data = ''
    
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)
            
            obs = ''
            base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
            ans = result['answerKey']
            choices = ''
            for j in range(len(result['question']['choices'])):
                choices = choices + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '

            obs =  base + choices + '[ANSWER]' + ans + ' <|endoftext|> '
            data = data + obs

        with open(f'{data_type}.txt', 'w') as f:
            f.write(data)

    def evaluate(self, model, data, tokenizer):
        model.eval()

        with torch.no_grad():

            # for calculating avg accuracy every N iterations
            correct_preds = 0

            for i in range(len(data)):
                obs = data[i]
                inputs = tokenizer(obs[:-1], truncation=True, return_tensors="pt")
                label = tokenizer(obs[-1], truncation=True, return_tensors="pt")['input_ids'][0][0]

                outputs = model(**inputs)
                pred_logits = outputs.logits[0][-1]
                vocab_probs = torch.softmax(pred_logits, dim=0)
                label_probs = [vocab_probs[self.label_inds[key]] for key in self.label_inds.keys()]

                pred_label_ind = torch.argmax(torch.tensor(label_probs), dim=0)
                pred_label = torch.tensor(list(self.label_inds.values()))[pred_label_ind]

                if pred_label == label:
                    correct_preds += 1
            
            accuracy = correct_preds/len(data)
            return accuracy
        
    def test(self, model_name):
        saved_model = GPT2LMHeadModel.from_pretrained(model_name)
        valid_acc = self.evaluate(saved_model, self.valid_set, self.tokenizer)
        test_acc = self.evaluate(saved_model, self.test_set, self.tokenizer)

        return valid_acc, test_acc

def create_plots():
    # Create plots from results.csv
    df = pd.read_csv('save/results.csv')
    plt.plot(df['train_accuracy'], label='train_acuracy')
    plt.plot(df['valid_accuracy'], label='valid_acuracy')
    plt.legend()
    plt.savefig('save/classifier_train_test_acc.png')
                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str,default='classifier')
    parser.add_argument('-mode', type=str,default='test')

    params = parser.parse_args()    

    if params.model == 'classifier':
        ### CLASSIFICATION ###
        classifier = Classify()

        if params.mode == 'train':
            classifier.train()
            create_plots()

        elif params.mode == 'test':
            valid_acc, test_acc = classifier.test('models/linear.pt')
            print(f"Fine-tuned CLASSIFIER Validation accuracy: {valid_acc} | Test accuracy: {test_acc}")

    elif params.model == 'generator':
        ### GENERATION ###

        generator = Generate()

        ### Clone https://github.com/huggingface/transformers and use the following script to fine-tune on data/train.txt:
        '''
        python3 -u transformers/examples/pytorch/language-modeling/run_clm.py \
                --model_name_or_path gpt2 \
                --train_file data/train.txt \
                --do_train \
                --output_dir models/gpt497 \
                --per_device_train_batch_size 2 \
                --num_train_epochs 5\
                >& output.log
        '''

        # Fine-tuned accuracy
        valid_acc, test_acc = generator.test("models/gpt497")
        print(f"Fine-tuned GENERATOR Validation accuracy: {valid_acc} | Test accuracy: {test_acc}")

if __name__ == "__main__": 
    main()