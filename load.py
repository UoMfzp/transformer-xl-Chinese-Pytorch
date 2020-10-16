# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:33:41 2020

@author: fzpef
"""


def get_train_step():
    try:
        with open("log/train_step", 'r') as f:
            train_steps = f.readlines()  
            for line in train_steps:
                train_step = float(line[0])
                return int(train_step)
    except ValueError as e:
        return 0
    
step=get_train_step()