# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:09:36 2020

@author: fzpef
"""

import argparse
import time
import math
import os, sys
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.data_parallel import BalancedDataParallel

parser = argparse.ArgumentParser(description='PyTorch TransformerXL Language Model')

parser.add_argument('--data', type=str, default='./data/text8',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='text8',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=10,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='LM-TFM',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
args = parser.parse_args()
args.tied = not args.not_tied

args.n_layer =16
args.d_model =410 
args.n_head =10
args.d_head =41
args.d_inner =1800 # inner dimension in FF
args.dropout =0.1
args.dropatt =0.0
args.lr =0.0003
args.warmup_step =0
args.max_step =100000
args.tgt_len =200
args.mem_len =400
args.eval_tgt_len =128
args.batch_size =32
args.gpu0_bsz=4 
args.d_embed=d_embed=410
is_train=False
is_infer=True

if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)
        
args.tied = not args.not_tied
device = torch.device('cuda' if args.cuda else 'cpu')

print(device)

###############################################################################
# Load data
###############################################################################
print("args.data=",args.data)
print("args.dataset",args.dataset)
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b']
    if args.dataset == 'wt103':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)



###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    #print("classname=",classname)
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)
            
def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

###############################################################################
# Build the model
###############################################################################
if is_train:
    args.restart=False
    if args.restart:
        #restart training from the saved checkpoint
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(weights_init)
        model.word_emb.apply(weights_init)
    #args.n_all_param = sum([p.nelement() for p in model.parameters()])
    #args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
    
    #半精度浮点型网络训练,模型和输入样本数据都要cuda().half()
    #当网络变大之后，半精度浮点型要比全精度浮点型要快
    if args.fp16:
        model = model.half()
    
    args.multi_gpu=False
    #多核模式 todo
    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)
    
    #### optimizer
    #### add new optimizers
    if args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #### scheduler
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                args.max_step, eta_min=args.eta_min) # should use eta_min arg
    
    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')



###############################################################################
# Training and evaluation
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    #不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, output, mems = ret[0], ret[1], ret[2:]   
            #loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len

def train():
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    #启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
        
    #count=0
    args.varlen=True
    #print("args.varlen=",args.varlen)
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    enTrIt = enumerate(train_iter)
    output = None
    for batch, (data, target, seq_len) in enTrIt:
        model.zero_grad()
        
        pre_out= output
        
        ret = para_model(data, target, *mems)
        loss, output, mems = ret[0], ret[1], ret[2:]        
        #print("   loss.shape   ",loss.shape)
        #print("   mems   ",np.array(mems).shape)
        loss = loss.float().mean().type_as(loss)
        print("loss={},epoch={},batch={},train_step={}".format(loss,epoch,batch,train_step))
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        train_loss += loss.float().item()
        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()
            
        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)    
         
        args.eval_interval=1    
        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                    save_train_step(train_step)
                best_val_loss = val_loss   
    return data, pre_out

def get_train_step():
    try:
        with open("log/train_step", 'r') as f:
            best_rate = f.readlines()  
            for line in best_rate:
                best_winrate = float(line[0])
                return best_winrate
    except ValueError as e:
        return 0

def save_train_step(train_step):
    with open("log/train_step","a+") as f:
        f.seek(0)
        f.truncate()
        f.writelines(train_step)

if is_train:
    # Loop over epochs.
    train_step = get_train_step()
    train_loss = 0
    best_val_loss = None
    #para_model = model.to(device)
    print("#####  enter train #######")
    # epochs = 1
    # for epoch in range(epochs):
    #     #train()
    #     data, output = train()
    #     #torch.save(para_model.state_dict(), "model/params.pkl")
    for epoch in itertools.count(start=1):
        data, output = train()
        if train_step == args.max_step:
            break

###############################################################################
# infer
###############################################################################
# store the train_steps
###############################################################################
def showList(infer_result):
    listCol=""
    #corpus.vocab.convert_to_sent([infer_result[:,0][0]])[0]
    for value in infer_result[:,0]:
        #print(value)
        listCol+=corpus.vocab.convert_to_sent([value])[0]+" "
    print("################# listCol ################")
    #print("len(output[:,0])={}".format(len(output[:,0])))
    print(listCol)

def showOutput(output):
    output=output.view(-1, 1, args.n_token)
    output_items=torch.tensor([[categoryFromOutput(output[i] for i in len(output))]])
    showList(output_items)
    #new_item=torch.tensor([[(categoryFromOutput(output[len(output)-1]))]])
    #print(output)
    # new_item=new_item.to(device)
    # print(new_item)   
    # test_start=torch.cat((test_start[1:len(test_start)],new_item),0)
    # test_start_target=test_start
    # infer_result=torch.cat((infer_result,new_item),0)

#result=corpus.vocab.convert_to_sent([3])
#output=output.view(-1, args.batch_size, args.n_token)
# test_start = torch.tensor([[item] for item in data[:,0]])
# print("test_start.shape",test_start.shape)
#test_start=data

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def infer():
    input_text="从这里开始"
    encoded_input = corpus.vocab.encode_sents(input_text, ordered=True)
    
    #test_start=[encoded_input]
    
    #test_start = torch.tensor([[item] for item in data[:,0]])
    test_start = torch.tensor([[item] for item in encoded_input])
    test_start = test_start.to(device)
    test_start_target = test_start
    
    test_time = 200
    mems = tuple()
    infer_result = test_start
    
    for time in range(test_time):
        ret = para_model(test_start, test_start_target, *mems, model_type="inferrence")
        print("test_time = {}".format(time))
        output, mems = ret[0], ret[1:]
        output=output.view(-1, 1, args.n_token)
        new_item=torch.tensor([[(categoryFromOutput(output[len(output)-1]))]])
        #print(output)
        new_item=new_item.to(device)
        #print(new_item)   
        test_start=torch.cat((test_start[1:len(test_start)],new_item),0)
        test_start_target=test_start
        # test_start=new_item
        # test_start_target=test_start
        infer_result=torch.cat((infer_result,new_item),0)
    
    listCol=""
    #corpus.vocab.convert_to_sent([infer_result[:,0][0]])[0]
    for value in infer_result[:,0]:
        #print(value)
        listCol+=corpus.vocab.convert_to_sent([value])[0]+" "
    print("################# listCol ################")
    print("len(output[:,0])={}".format(len(output[:,0])))
    print(listCol)

if is_infer:
    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    para_model = model.to(device)
    infer()




# listCol=""
# for value in output[:,0]:
#     listCol+=corpus.vocab.convert_to_sent([(categoryFromOutput(value))])[0]+" "
# print("################# listCol ################")
# print("len(output[:,0])={}".format(len(output[:,0])))
# print(listCol)

# listRow=""
# for value in output[0,:]:
#     listRow+=corpus.vocab.convert_to_sent([(categoryFromOutput(value))])[0]+" "
# print("################# listCol ################")
# print("len(output[:,0])={}".format(len(output[0,:])))
# print(listRow)

# result1=corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[0]])
# result2=corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[1]])
# result3=corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[2]])
# result4=corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[3]])
# result5=corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[4]])

# result=[]
# i=0
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[0]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[1]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[2]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[3]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[4]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[5]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[6]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[7]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[8]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[9]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[10]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[11]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[12]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[13]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[14]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[15]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[16]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[17]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[18]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[19]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[20]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[21]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[22]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[23]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[24]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[25]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[26]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[27]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[28]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[29]])[i])
# result.append(corpus.vocab.convert_to_sent([categoryFromOutput(i) for i in output[30]])[i])
# print(result)
