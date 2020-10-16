# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:21:48 2020

@author: fzpef
"""
import torch
import getData

# def novel_generator(model, TEXT):
#     length=50
#     start_item=torch.tensor([[1]])
#     input=start_item
#     resultVecs=[]
#     for i in range(length):
#         output=model(input)
#         top_n, top_i = output.topk(1)
#         resultVecs.append(top_i)
#         input=top_i.squeeze(-1)
#     print(getData.vectorsToWords(resultVecs,TEXT))
    
def novel_generator(model, start_data, TEXT):
    length=50
    start_item=torch.tensor([[1]])
    input=start_item
    resultVecs=[]
    for i in range(length):
        output=model(input)
        top_n, top_i = output.topk(1)
        resultVecs.append(top_i)
        input=top_i.squeeze(-1)
    print(getData.vectorsToWords(resultVecs,TEXT))
