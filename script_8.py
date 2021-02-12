from __future__ import division
import argparse
from time import strftime
#from BTC_encoder import encoder_class , generate_constellation

from multiprocessing import Pool

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import gc

import torch.distributions as distributions
#torch.autograd.set_detect_anomaly(True)
from profiler_module import profiler_class
from tqdm import tqdm

torch.manual_seed(123)
np.random.seed(123)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using sigma value for 1 dimensional constellation.')

def noise_sigma(snr_db):
    
    snr_db      =   snr_db + 10*np.log10(rate*1) # bpsk has 2**1 points in constellation
    snr_val     =   10**(snr_db/10)
    sigma_val   =   np.sqrt(args.power/snr_val)
    
    return sigma_val

def find_correct(out_dec,messages):
    
    out_dec     =   out_dec.detach()
    messages    =   messages.detach()
    
    out_dec     =   out_dec.round()
    
    diff        =   out_dec - messages
    diff_abs    =   1.0 - diff.abs()
    
    sum_diff    =   diff_abs.sum()
    
    return sum_diff

class encoder_small():
    
    def __init__(self,H,G):
    
       # H small is with the m,n matrix which is small in size 
       # this is only used with one particular H matrix for others
       # we could have different way of encoding.
        
        self.H          =   H
        self.G          =   G
        self.m,self.n   =   self.H.shape
        self.k          =   self.n - self.m
        
        # H =   [P, Im]
        # G =   [Ik -P.t]
        
    def __call__(self,msg):
        
        cwd     =   np.matmul(msg,self.G) % 2
        
        cwd     =   2.0*cwd-1.0
        cwd     =   cwd*np.sqrt(args.power)
        
        return cwd

class encoder():
    
    def __init__(self,H,G):
        
        self.encoder_small_obj  =   encoder_small(H,G)
    
    def __call__(self,msg):
        
        pool_obj = Pool()
        cwd         =   pool_obj.map(self.encoder_small_obj,msg)
        pool_obj.close()
        del pool_obj
        
        cwd         =   np.array(cwd)
        
        cwd         =   torch.tensor(cwd).double().to(device)
        return cwd


class graph_structure():
    
    def __init__(self,m,n,H):
        self.m  =   m
        self.n  =   n
        self.H  =   H
        
        self.edge_set   =   {}
        self.v_edge_set =   {}
        self.c_edge_set =   {}
        
        for j in range(self.n):
            self.v_edge_set[j]  = []
            
        for i in range(self.m):
            self.c_edge_set[i]  = []   
        
        edge_no           =   0
        
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i,j] == 1:
                    self.edge_set[edge_no] = [i,j]
                    self.v_edge_set[j].append(edge_no)
                    self.c_edge_set[i].append(edge_no)
                    edge_no = edge_no+1
                    
        self.dvi        =   self.H.sum(axis=0)
        self.dci        =   self.H.sum(axis=1)
        self.total_edges=   self.H.sum()
        
        self.v_c_edges =   {}
        for index, v_connections in self.v_edge_set.items():
            set_connections =   set(v_connections)
            for edge_no in v_connections:
                self.v_c_edges[edge_no]    =   set_connections - {edge_no}
        
        self.v_sum_indices   =   []
        
        for index in range(self.total_edges):
            set_connections         =   self.v_c_edges[index]
            list_connections        =   list(set_connections)
            self.v_sum_indices      =   self.v_sum_indices + list_connections
        
        
        self.c_v_edges =   {}
        for index, c_connections in self.c_edge_set.items():
            set_connections =   set(c_connections)
            for edge_no in c_connections:
                self.c_v_edges[edge_no]    =   set_connections - {edge_no}
        
        self.c_prod_indices   =   []
        
        for index in range(self.total_edges):
            set_connections         =   self.c_v_edges[index]
            list_connections        =   list(set_connections)
            self.c_prod_indices     =   self.c_prod_indices + list_connections
        
        self.total_weights          =   len(self.v_sum_indices)
        
        self.ind_vsum               =   []
        temp                        =   0
        self.ind_vsum.append(temp)
        
        for index in range(self.total_edges):
            v_num   =   self.edge_set[index][1]
            temp    =   temp + self.dvi[v_num] -1
            self.ind_vsum.append(temp)
        
        self.ind_cprod              =   []
        temp                        =   0
        self.ind_cprod.append(temp)
        
        for index in range(self.total_edges):
            c_num   =   self.edge_set[index][0]
            temp    =   temp + self.dci[c_num]-1
            self.ind_cprod.append(temp)
        
        self.ind_final_vsum     =   []
        self.marginalize_ind    =   []
        temp                    =   0
        self.marginalize_ind.append(temp)
        
        for j in range(self.n):
            edge_set_j  =   self.v_edge_set[j]
            self.ind_final_vsum = self.ind_final_vsum + edge_set_j 
            temp        =   len(edge_set_j)
            self.marginalize_ind.append(temp)
        
#       Here is print block for cross checking of the edge connections.
        
#        print(self.v_sum_indices)
#        
#        for index in range(self.total_edges):
#            v_num   =   self.edge_set[index][1]
#            print(index,self.v_c_edges[index])
#            print(self.dvi[v_num]-1)
#        
#        
#        print(self.c_prod_indices)
#        
#        for index in range(self.total_edges):
#            c_num   =   self.edge_set[index][0]
#            print(index,self.c_v_edges[index])
#            print(self.dci[c_num]-1)
        
    def get_edge_indices(self):
        return self.v_sum_indices, self.c_prod_indices
        
    def get_ind_vsum_cprod(self):
        return self.ind_vsum, self.ind_cprod
        
    def get_num_weights(self):
        return self.total_weights
    
    def get_edge_set(self):
        return self.edge_set
    
    def get_ind_final_vsum(self):
        return self.ind_final_vsum, self.marginalize_ind
    

def get_args(): 
    
    parser      =   argparse.ArgumentParser()
    
    parser.add_argument('--exp-dir',type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--n-iteration',type = int ,default = 5)
    
    parser.add_argument('--test-snr-low',type=float,default =2.0)
    parser.add_argument('--test-snr-high',type=float,default=6.0)
    parser.add_argument('--test-snr-step',type=float,default=1.0)
    
    parser.add_argument('--numblocks',type=int,default =10000)
    parser.add_argument('--batch-size',type=int,default=1000)
    
    parser.add_argument('--test-numblocks',type=int,default =10000)
    parser.add_argument('--test-batch-size',type=int,default=1000)
    
    parser.add_argument('--check-every',type=int,default=10)
    
    parser.add_argument('--M', type=int, default=2)
    
    parser.add_argument('--power',type=float,default =1.0)
    
    parser.add_argument('--train-snr',type=float,default=4, help='All SNR in EsN0.')
    parser.add_argument('--training-snr-low',type=float,default = 0.0)
    parser.add_argument('--training-snr-high',type=float,default= 8.0)
    parser.add_argument('--training-type',type=str,default='fix')  #'range','fix')
    parser.add_argument('--arch-disc',type=str,default='ldpc_bp')
    parser.add_argument('--test-in-range',type=bool,default = True)
    
    args = parser.parse_args()
    
    return args

class decoder(nn.Module):
    
    def __init__(self,m,n,H,iterations):
        super(decoder,self).__init__()

        self.m          =   m
        self.n          =   n
        self.H          =   H
        
        self.tanner_graph   =   graph_structure(self.m,self.n,self.H)
        self.v_sum_indices, self.c_prod_indices =   self.tanner_graph.get_edge_indices()
        self.ind_vsum, self.ind_cprod           =   self.tanner_graph.get_ind_vsum_cprod()
        self.total_weights                      =   self.tanner_graph.get_num_weights()
        self.edge_set                           =   self.tanner_graph.get_edge_set()
        self.ind_final_vsum, self.marginalize_ind =   self.tanner_graph.get_ind_final_vsum()
        
        self.v_sum_indices  =   torch.tensor(self.v_sum_indices).long().to(device)
        self.c_prod_indices =   torch.tensor(self.c_prod_indices).long().to(device)
        self.ind_final_vsum =   torch.tensor(self.ind_final_vsum).long().to(device)
        
        self.total_edges    =   self.H.sum()
        
        self.iterations =   iterations
        
        self.sigmoid        =   nn.Sigmoid()
        self.tanh           =   nn.Tanh()
        self.w_param        =   nn.ParameterDict()
        self.llr_param      =   nn.ParameterDict()
        
        for iter_idx in range(self.iterations):
            
            self.w_param['iter_{}_edge_params'.format(iter_idx)]  = nn.Parameter(torch.ones(self.total_weights).double().to(device),requires_grad = True)
            self.llr_param['iter_{}_llr_params'.format(iter_idx)] = nn.Parameter(torch.ones(self.n).double().to(device),requires_grad = True)
                
        
        self.w_param['final_edge_params']   = nn.Parameter(torch.ones(self.total_edges).double().to(device),requires_grad = True)
        self.llr_param['final_llr_params']  = nn.Parameter(torch.ones(self.n).double().to(device),requires_grad = True)
        
    
    def var_to_check_calc(self,check_to_var_msg,llr,iter_idx):
        
        batch_size  =   check_to_var_msg.shape[0]
        sz          =   self.v_sum_indices.shape[0]
        indices     =   self.v_sum_indices.unsqueeze(0).expand(batch_size,sz)
        
        check_to_var_sum_sets   =   torch.gather(check_to_var_msg,1,indices)
        weighted_check_var_sets =   check_to_var_sum_sets*self.w_param['iter_{}_edge_params'.format(iter_idx)]
        weighted_llr            =   llr*self.llr_param['iter_{}_llr_params'.format(iter_idx)]
        
        weighted_check_var      =   []
        
        for i in range(self.total_edges):
            i1      =   self.ind_vsum[i]
            i2      =   self.ind_vsum[i+1]
            llr_no  =   self.edge_set[i][1]
            message_to_edge_i   =   weighted_check_var_sets[:,i1:i2]
            message_to_edge_i   =   message_to_edge_i.sum(dim=1)
            message_to_edge_i   =   message_to_edge_i.unsqueeze(1)
            message_to_edge_i   =   message_to_edge_i + weighted_llr[:,llr_no].unsqueeze(1)
            weighted_check_var.append(message_to_edge_i)
        
        weighted_check_var      =   torch.cat(weighted_check_var,dim=1)
        
        var_to_check_msg        =   self.tanh(0.5*weighted_check_var)
        
        return var_to_check_msg
    
    def check_to_var_calc(self,var_to_check_msg,iter_idx):
        
        batch_size  =   var_to_check_msg.shape[0]
        sz          =   self.c_prod_indices.shape[0]
        indices     =   self.c_prod_indices.unsqueeze(0).expand(batch_size,sz)
        
        var_to_check_prod_sets  =   torch.gather(var_to_check_msg,1,indices)
        
        msg_sets                =   []
        
        for i in range(self.total_edges):
            i1      =   self.ind_cprod[i]
            i2      =   self.ind_cprod[i+1]
            message_to_edge_i   =   var_to_check_prod_sets[:,i1:i2]
            message_to_edge_i   =   message_to_edge_i.prod(dim=1)
            message_to_edge_i   =   message_to_edge_i.unsqueeze(1)
            msg_sets.append(message_to_edge_i)
        
        msg_sets      =   torch.cat(msg_sets,dim=1)
        
        check_to_var_msg    =   2*torch.atanh(0.999995*msg_sets)   
        
        return check_to_var_msg
    
    def final_output_calc(self,check_to_var_msg,llr):
        
        batch_size  =   check_to_var_msg.shape[0]
        sz          =   self.ind_final_vsum.shape[0]
        indices     =   self.ind_final_vsum.unsqueeze(0).expand(batch_size,sz)

        check_to_var_sum_sets   =   torch.gather(check_to_var_msg,1,indices)
        weighted_check_var_sets =   check_to_var_sum_sets*self.w_param['final_edge_params']
        weighted_llr            =   llr*self.llr_param['final_llr_params']
        
        final_output      =   []
        
        for i in range(self.n):
            i1      =   self.marginalize_ind[i]
            i2      =   self.marginalize_ind[i+1]
            llr_no  =   i
            message_to_edge_i   =   weighted_check_var_sets[:,i1:i2]
            message_to_edge_i   =   message_to_edge_i.sum(dim=1)
            message_to_edge_i   =   message_to_edge_i.unsqueeze(1)
            message_to_edge_i   =   message_to_edge_i + weighted_llr[:,llr_no].unsqueeze(1)
            final_output.append(message_to_edge_i)
        
        final_output      =   torch.cat(final_output,dim=1)
        
        return final_output
    
    def forward(self,llr):
        
#        print(llr)
        
        # Zero belief from check nodes in the begging.
        check_to_var_msg        =   torch.zeros(llr.shape[0],self.total_edges).double().to(device)
        
#        print(check_to_var_msg)
        for iter_idx in range(self.iterations):
            var_to_check_msg    =   self.var_to_check_calc(check_to_var_msg,llr,iter_idx)
#            print(var_to_check_msg)
            check_to_var_msg    =   self.check_to_var_calc(var_to_check_msg,iter_idx)
#            print(check_to_var_msg)
        
        final_output    =   self.final_output_calc(check_to_var_msg,llr)
        
#        print(final_output)
        
        output          =   final_output[:,:self.n-self.m]
        
        output          =   self.sigmoid(output)
        
        return output


def train(epoch_number):
    
    num_batches        =   int(args.numblocks/args.batch_size)
    
    dec_obj.train()
    
    train_loss          =   0.0
    correct             =   0.0
    
    for idx in range(num_batches): 
        
        random_snr      =   np.random.uniform(args.training_snr_low,args.training_snr_high)
        if args.training_type == 'range':
            sigma       =   noise_sigma(random_snr)
        else:
            sigma           =   noise_sigma(args.train_snr)
        
        noise_generator =   distributions.normal.Normal(0, sigma)
            
        optimizer_dec.zero_grad()
        
        messages        =   np.random.randint(0,2,(args.batch_size,k))
        
        out_enc         =   enc_obj(messages)
        out_enc         =   out_enc.double().to(device)
        
        messages        =   torch.tensor(messages).double().to(device)
            
        noise           =   noise_generator.sample(out_enc.shape).to(device).double()
        out_channel     =   out_enc + noise

        div             =   sigma**2
        llr             =   2*out_channel/div 
        out_dec         =   dec_obj(llr)
        
        loss            =   bce_loss(out_dec,messages)
        
        temp_loss       =   loss.item()
        temp_correct    =   find_correct(out_dec,messages)
        
        loss.backward()
        optimizer_dec.step()
            
        del loss
        train_loss                     += temp_loss  # loss.item()
        correct                        +=temp_correct
        
    
    train_acc       =   float(correct)/(k*num_batches*args.batch_size)
    print('train_acc = ',train_acc)
    train_ber       =   1.0-train_acc
    
    str_print       =   "\nEpoch Number {:}: \t Train Loss = {:}, \t\t Train BER = {:}".format(epoch_number,train_loss,train_ber)
    
    print(str_print)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_loss, train_ber



def test(epoch_number,noise_generator= None):
    
    dec_obj.eval()
    
    sigma           =   noise_sigma(args.train_snr)
    if noise_generator is None:
        noise_generator =  distributions.normal.Normal(0, sigma)
    else:
        sigma = noise_generator.scale
      
    test_loss           =   0.0
    correct             =   0.0
    
    num_batches         =   1 # int(args.test_numblocks/args.test_batch_size)
    
#    print('Using sigma value as ' ,sigma)
    with torch.no_grad():
            
        messages            =   np.random.randint(0,2,(args.test_batch_size,3))
        out_enc             =   enc_obj(messages)
        
        out_enc             =   out_enc.double().to(device)
        messages            =   torch.tensor(messages).double().to(device)
            
        noise               =   noise_generator.sample(out_enc.shape).to(device).double()
        out_channel         =   out_enc + noise
        
        div                 =   sigma**2
        llr                 =   2*out_channel/div 
        out_dec             =   dec_obj(llr)
            
        loss                =   bce_loss(out_dec, messages) 
            
        test_loss          +=   loss.item()
        temp_correct        =   find_correct(out_dec,messages)
        correct            +=   temp_correct
            
#    print('value of correct is : ',correct)
#    print('divisor is : ',args.k1 * args.batch_size * num_batches)
            
    test_acc            =   float(correct)/(k*num_batches*args.test_batch_size)
#    print('train_acc :',train_acc)
    test_ber            =   1.0-test_acc
    str_print           =   "\n                  \t Test  Loss = {:f}, \t\t Test BER  = {:}\n".format(test_loss,test_ber)
    print(str_print)
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_loss, test_ber

def test_in_range():
    
    l       =   args.test_snr_low
    h       =   args.test_snr_high + args.test_snr_step
    step    =   args.test_snr_step
    
    test_ber_vals       =   []
    test_snr_vals       =   []
    
    for snr in np.arange(l, h, step):
        
        noise_generator =     distributions.normal.Normal(0, noise_sigma(snr))
        
        __, ber_val     =   test(epoch,noise_generator)
        
        test_ber_vals.append(ber_val)
        test_snr_vals.append(snr)
    
    test_ber_vals       =   np.array(test_ber_vals)
    test_snr_vals       =   np.array(test_snr_vals)
    
    filename            =   args.exp_dir
    
    if not os.path.isdir(filename):
        os.makedirs(filename)
    
    write_these                 =   {}
    write_these['SNR_value ']   =   test_snr_vals
    write_these['BER_value ']   =   test_ber_vals
    
    filename        =   os.path.join(filename,'range_test.csv')
    rng_test_file   =   open(filename,'w+')
    
    df=pd.DataFrame(data=write_these,dtype=np.float32)
    df.to_csv(rng_test_file)
    rng_test_file.close()
    
    return test_snr_vals, test_ber_vals


if __name__ == '__main__':
    
    device_ids      =   [0,1,2,3]
    
    args            =   get_args()
    profiler        =   profiler_class(args)
    
    assert args.numblocks       ==  args.test_numblocks
    assert args.test_batch_size ==  args.batch_size
    
    H = np.array([[1,1,1,1,0,0],
                  [0,1,1,0,1,0],
                  [1,0,1,0,0,1]])
    
    G = np.array([[1,0,0,1,0,1],
                  [0,1,0,1,1,0],
                  [0,0,1,1,1,1]])
    
    m   =   H.shape[0]
    n   =   H.shape[1]
    
    enc_obj         =   encoder(H,G)     
    
    dec_obj         =   decoder(m,n,H,args.n_iteration).double().to(device)
    
    k               =   n-m
    
    rate            =   k/n
    
    bce_loss        =   nn.BCELoss()
    
    
    lr              =   args.lr
    check_every     =   args.check_every
    curr_checkpoint =   0
    
    training_ber_curve  =   []
    testing_ber_curve   =   []
    
    print(list(dec_obj.parameters()))
    
    try:
        for epoch in range(1, args.epochs + 1):
        
#            lr              =   lrscheduler(epoch)
            optimizer_dec   =   optim.Adam(list(dec_obj.parameters()), lr=lr)
            
            __, trainber    =   train(epoch)
            __, testber     =   test(epoch)
            
            training_ber_curve.append(trainber)
            testing_ber_curve.append(testber)
            
            if epoch % check_every == 0:
                curr_checkpoint = curr_checkpoint + 1
                profiler.logs(enc_obj,dec_obj,test_in_range,curr_checkpoint)
            
            if epoch % 5 ==0 :
                profiler.save_learning_graphs(training_ber_curve,testing_ber_curve)
            
            gc.collect()
            
    except KeyboardInterrupt:
        print('KeyboardInterrupt caught. !!! Exiting the program.')
    
    if curr_checkpoint <= int(args.epochs/check_every):
        profiler.logs(enc_obj,dec_obj,test_in_range,curr_checkpoint+1)
    
    profiler.final_logs(training_ber_curve,testing_ber_curve)
