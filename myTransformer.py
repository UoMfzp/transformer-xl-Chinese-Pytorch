# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 08:05:51 2020

@author: fzpef
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import getData
import model
import generator

def train(epoch):
    model.train()
    total_loss=0.
    start_time=time.time()
    ntokens=len(TEXT.vocab.stoi)
    #for batch,i in enumerate(range(0, train_data.size(0)-1,bptt)):
    for batch,i in enumerate(range(0, bptt-1,bptt)):
        data,targets=getData.get_batch(train_data,i,bptt)
        
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output.view(-1,ntokens),targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),0.5)
        optimizer.step()
        
        total_loss += loss.item()
        log_interval = 6
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('loss {:2.2f} | epoch {:2d}'.format(cur_loss, epoch))
            # print('|epoch {:3d}|{:5d}/{:5d} batches|'
            #       'lr{:02.2f}|ms/batch{:5.2f}| '
            #       'loss {:5.2f}|ppl{:8.2f}'.format(
            #         epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
            #         elapsed * 1000 / log_interval,
            #         cur_loss, math.exp(cur_loss)))
            # print('|epoch {:3d}|{:5d}/{:5d} batches|'
            #       'lr{:02.2f}|ms/batch{:5.2f}| '
            #       'loss {:5.2f}|ppl{:8.2f}'.format(
            #         epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
            #         elapsed * 1000 / log_interval,
            #         cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

############### main #################

batch_size=20
eval_batch_size=10
#train_data,val_data,test_data,TEXT,device = getData.getTrainValTest(batch_size, eval_batch_size)
train_data,start_data,TEXT,device = getData.get_train_data(batch_size, eval_batch_size)

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 512 # embedding dimension
nhid = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = model.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 3
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)
bptt = 62 # length of sequence 

if 1:
    print("############ train ###########")
    epochs = 1
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print("############ new epoch ###########")
        train(epoch)
        scheduler.step()
    # generator.novel_generator(model, TEXT)
    torch.save(model, "novelModel.pkl")
    
    # length=1
    # start_item=torch.tensor([[1]])
    # print(start_data)
    # input=start_data.unsqueeze(-1)
    # resultVecs=[]
    # for i in range(length):
    #     output=model(input)
    #     resultVecs=getData.outputToVecs(output)
    # print(getData.vectorsToWords(resultVecs,TEXT))
    
else:
    model = torch.load("novelModel.pkl")
    #generator.novel_generator(model, start_data, TEXT)
    
    length=1
    start_item=torch.tensor([[1]])
    print(start_data)
    input=start_data.unsqueeze(-1)
    resultVecs=[]
    
    for i in range(length):
        output=model(input)
        resultVecs=getData.outputToVecs(output)

    print(getData.vectorsToWords(resultVecs,TEXT))


#print(getData.vectorsToWords(start_data,TEXT))
#print(getData.vectorsToWords(start_data,TEXT))

# model = torch.load("novelModel.pkl")
# start=getData.get_start(train_data, bptt)
# #generator.novel_generator(model, start, TEXT)
# length=50
# start_item=torch.tensor([[1]])
# input=start_item
# resultVecs=[]
# output=model(start_item)

# for i in range(length):
#     output=model(start)
#     top_n, top_i = output.topk(1)
#     resultVecs.append(top_i)
#     input=top_i.squeeze(-1)
# print(getData.vectorsToWords(resultVecs,TEXT))
