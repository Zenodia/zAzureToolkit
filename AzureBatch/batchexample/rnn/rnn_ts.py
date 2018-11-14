from __future__ import unicode_literals, print_function, division
import os
import argparse
import pandas as pd
import azure.storage.blob as azureblob
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib as plt
import torch
from torch.autograd import Variable


if __name__ == "__main__":
    # Example Syntax
    # python .\batch\make_lm.py -i .\data\split\splitaustralia.csv --storageaccount azueus2devsabatch --storagecontainer output --key EBVkUou5ePWV0qvGqQtrCHgzuk45ltfdExylQQLkGGFYUgXaERZJUyY2UGq2opwgs4wj9XuJGtKVZlY6BhtN8Q==
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', help = "The location of the input file", required = True)
    
    parser.add_argument('--storageaccount', required=True,
                        help='The name the Azure Storage account that owns the blob storage container to which to upload results.')
    parser.add_argument('--storagecontainer', required=True,
                        help='The Azure Blob storage container')
    parser.add_argument('--key', required=True,
                        help='The access key providing write access to the Storage container.')
    
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile,sep=';')
    df['Month of Year']=df['Month of Year'].astype("str")
    df['month']=df['Month of Year'].apply(lambda x: x[4:])
    df['year']=df['Month of Year'].apply(lambda x: x[:4])
    #construct cuts to make the symbols
    cuts=[df['Sessions'].quantile(.1*i) for i in range(0,10)]
    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Lowercase, trim, and remove non-letter characters
    
    
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    # convert Session data to strings as symbols 
    symbols=[]
    for idx in df.index:
        if df.iloc[idx,1]< cuts[0]:
            #using normalizeString to get rid of unicode u'xx 
            symbols.append(normalizeString('a'))
        elif (df.iloc[idx,1]>=cuts[0] and df.iloc[idx,1] < cuts[1]):
            symbols.append(normalizeString('b'))
        elif (df.iloc[idx,1]>=cuts[1] and df.iloc[idx,1] < cuts[2]):
            symbols.append(normalizeString('c'))
        elif (df.iloc[idx,1]>=cuts[2] and df.iloc[idx,1] < cuts[3]):
            symbols.append(normalizeString('d'))
        elif (df.iloc[idx,1]>=cuts[3] and df.iloc[idx,1] < cuts[4]):
            symbols.append(normalizeString('e'))
        elif (df.iloc[idx,1]>=cuts[4] and df.iloc[idx,1] < cuts[5]):
            symbols.append(normalizeString('f'))
        elif (df.iloc[idx,1]>=cuts[5] and df.iloc[idx,1] < cuts[6]):
            symbols.append(normalizeString('g'))
        elif (df.iloc[idx,1]>=cuts[6] and df.iloc[idx,1] < cuts[7]):
            symbols.append(normalizeString('h'))
        elif (df.iloc[idx,1]>=cuts[7] and df.iloc[idx,1] < cuts[8]):
            symbols.append(normalizeString('i'))
        elif (df.iloc[idx,1]>=cuts[8] and df.iloc[idx,1] < cuts[9]):
            symbols.append(normalizeString('j'))
        else :
            symbols.append(normalizeString('k'))
    #print(len(symbols),symbols)
    df['symbols']=symbols
    print(df.head())
    lookup_symbols=dict(zip(df['Sessions'],df['symbols']))
    train_data=[]

    yrs=df['year'].unique().tolist()
    for yr in yrs:    
        temp=[lookup_symbols[x] for x in df.loc[(df.year==str(yr))]['Sessions'].values.tolist()]
        train_data.append(temp)
    
    word_to_ix = {}
    for sent in train_data:
        
        for word in sent:
            if str(word) not in word_to_ix:
                word_to_ix[str(word)] = len(word_to_ix)
    #print(word_to_ix)
    #construct target for supervised training
    target=[]
    for item in train_data:
        if 'i' in item:
            target.append([1])
        else:
            target.append([0])
    print(target)
    print(target[-1][0])
    
    ########## RNN Model (Many-to-One)
    # Hyper Parameters
    sequence_length = 12
    input_size = 12
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 1
    num_epochs = 100
    learning_rate = 0.01
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # Set initial states 
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            out, _ = self.lstm(x, (h0, c0))  
            
            # Decode hidden state of last time step
            out = self.fc(out[:, -1, :])  
            return out
    
    rnn = RNN(input_size, hidden_size, num_layers, num_classes)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    out_loss=[]
    # Train the Model
    for epoch in range(num_epochs):
    
        for i in range(len(train_data)-1): # len(train_data)-1 is to hold out last year
            optimizer.zero_grad()
            tryin=[word_to_ix[x] for x in list(train_data[i])] 
            input_seq= Variable(torch.FloatTensor([tryin])).unsqueeze(0)
            output_seq = rnn(input_seq)
            last_output = output_seq
    
            err = criterion(last_output, Variable(torch.LongTensor(target[i])))
            err.backward()
            optimizer.step()
        if epoch%10 ==0 :
            print("loss",err)
            out_loss.append((epoch,err))
        
        # Test the Model with last years data that we hold out
    # Test the Model with the one year 2015 that we hold out
    correct=0
    incorrect=0
    predictions=[]
    for i in range(len(train_data)): # len(train_data)-1 is to hold out last year
    
        tryin=[word_to_ix[x] for x in list(train_data[i])] 
    
        print(tryin)
        input_seq= Variable(torch.FloatTensor([tryin])).unsqueeze(0)
        output_seq = rnn(input_seq)
        _, pred = torch.max(output_seq.data, 1)
        print("pred",pred)
        print(type(pred))
        
        if pred[0]==target[-1][0]:
            print("correct")
            correct+=1
        else:
            print("incorrect")
            incorrect+=1
    print(correct/(correct+incorrect))
    print(incorrect/(correct+incorrect))
    
    output_file_1 = "{}_RNN_many2one.pt".format(os.path.splitext(args.inputfile)[0])
    output_file_2 = "{}_RNN_performance.txt".format(os.path.splitext(args.inputfile)[0])

    torch.save(rnn.state_dict(), output_file_1)
    the_model = RNN(input_size, hidden_size, num_layers, num_classes)
    #the_model.load_state_dict(torch.load(output_file_1))
    with open(output_file_2, 'w') as out:
        out.write("Model performance info for {}\n".format(args.inputfile))
        out.write("correct :{}, incorrect:{}".format(correct/len(target),incorrect/len(target)))
    
    
    def write_to_blob(output_file_path, output_file):    

        blob_client = azureblob.BlockBlobService(account_name=args.storageaccount, account_key=args.key)
    
        blob_client.create_blob_from_path(args.storagecontainer,output_file,output_file_path)
    
    output_file_path_1 = os.path.realpath(output_file_1)
    write_to_blob(output_file_path_1,output_file_1 )
    output_file_path_2 = os.path.realpath(output_file_2)
    write_to_blob(output_file_path_2,output_file_2 )
    

    