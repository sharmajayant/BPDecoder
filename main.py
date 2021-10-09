from __future__ import division
import argparse
from time import strftime
from multiprocessing import Pool

from BCH_code_library import get_bch_G_H_mat
from build_graph import graph_structure
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

#torch.manual_seed(123)
#np.random.seed(123)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class encoder_class():
    
    def __init__(self,n,k,generator_mat):
        
        self.k, self.n          =   k, n 
        self.generator_mat      =   generator_mat
        
    def single_cpu_encoder(self,single_msg):
        
        codeword        =   np.matmul(single_msg,self.generator_mat) %2
        
        return codeword
        
    def __call__(self,msg):
        
#        msg         =   np.array(msg)
#        print('shape of message is',msg.shape)
        
        pool_obj        =   Pool()
        all_codewords   =   pool_obj.map(self.single_cpu_encoder,msg)
        pool_obj.close()
        del pool_obj
        
        all_codewords   =   np.array(all_codewords)
        
        all_codewords   =   torch.tensor(all_codewords).to(device).double()
        
        return all_codewords
    

class channel_model():
    
    def __init__(self,noise_type='awgn',choice='bpsk',snr_db=0.0,rate=1,ebno_esno='ebno',device='cuda'):
        
        assert (choice == 'qpsk') or (choice == 'bpsk')
        
        num_points = 2 if choice == 'bpsk' else 4  
        
        inverse_gray_map    =   [0,1,3,2]
        gray_map            =   [0,1,3,2]
        
        if choice   == 'bpsk':
            const_points = np.array([ -1.0, 1.0])  
        else:
            const_points    =   np.zeros((num_points,2),dtype = np.float)
            phase_step      =   2*np.pi/num_points
            
            for i in range(num_points):
                curr_phase          = phase_step*float(gray_map[i]) + float(phase_step/2)
                xval                = np.cos(curr_phase)
                yval                = np.sin(curr_phase)
                const_points[i,0]   = xval
                const_points[i,1]   = yval
        
        self.gray_map           =   gray_map
        self.inverse_gray_map   =   inverse_gray_map
        self.num_points         =   num_points
        self.choice             =   choice
        self.const_points       =   torch.tensor(const_points).to(device).double()
        self.mod_dimension      =   1 if choice == 'bpsk' else 2
        self.noise_type         =   noise_type
        self.ebno_esno          =   ebno_esno
        self.rate               =   rate
        self.device             =   device
        self.snr_db             =   snr_db
    
    def __str__(self):
        string_to_return        =   "Channel: {}, snr_db: {}, mod_type: {}, snr_type: {}, rate: {}".format(self.noise_type, self.snr_db, self.choice, self.ebno_esno, self.rate)
        
        return string_to_return
        
    def noise_sigma(self,snr_db):
        
        if self.ebno_esno == 'ebno':
            M           =   np.log2(self.num_points)
            snr_db      =   snr_db + 10*np.log10(self.rate*M) # bpsk has 2**1 points in constellation
        
        sigma_val    =   (1.0/np.sqrt(2))*(10**(-snr_db/20))
        
        return sigma_val   
        
    def modulate(self,cwd_tensor):
        
        if self.mod_dimension ==1 :
            output_tensor        =   2*cwd_tensor-1.0
        else:
            shape               =   cwd_tensor.shape
            cwd_tensor          =   cwd_tensor.reshape(-1,2)
            cwd_tensor          =   2*cwd_tensor[:,0] + cwd_tensor[:,1]
            cwd_tensor          =   cwd_tensor.long()
            
            cwd_tensor          =   cwd_tensor.unsqueeze(1)
            cwd_tensor          =   torch.cat([cwd_tensor,cwd_tensor],dim=1)
            
            output_tensor       =   torch.gather(self.const_points,dim=0,index=cwd_tensor)
            output_tensor       =   output_tensor.reshape(shape[0],-1,2)
            
        output_tensor       =   output_tensor.to(self.device).double()
        
        return output_tensor
    
    def add_noise(self,modulated_tensor):
        
        if len(self.snr_db) == 2:
            lower   =   self.snr_db[0]
            upper   =   self.snr_db[1]
            mult    =   upper-lower
            current =   mult*np.random.rand() + lower
            snr     =   current
            sigma   =   self.noise_sigma(snr)
        else:
            snr     =   self.snr_db[0]
            sigma   =   self.noise_sigma(snr)
        
        if self.noise_type == 'awgn':
            self.noise_generator    =  distributions.normal.Normal(0, sigma)
        else:
            self.noise_generator    =   None
            pass
        
        noise                   =   self.noise_generator.sample(modulated_tensor.shape)
        noise                   =   noise.to(self.device).double()
        noisy_tensor            =   modulated_tensor + noise
        
        return noisy_tensor, sigma
    
    def demodulate(self,noisy_tensor,sigma):
        
        shape                   =   noisy_tensor.shape
        if self.choice == 'bpsk':
            llr_values          =   2*noisy_tensor/(sigma**2)  # 2r/(sigma**2)
        else:
            
            sigma_sq            =   sigma**2
            batch_size          =   noisy_tensor.shape[0]
            channel_output      =   noisy_tensor
            
            d0                  =   (channel_output - self.const_points[0,:])**2
            d1                  =   (channel_output - self.const_points[1,:])**2
            d2                  =   (channel_output - self.const_points[2,:])**2
            d3                  =   (channel_output - self.const_points[3,:])**2
            
            p0                  =   torch.exp(-d0.sum(dim=2)/sigma_sq)
            p1                  =   torch.exp(-d1.sum(dim=2)/sigma_sq)
            p2                  =   torch.exp(-d2.sum(dim=2)/sigma_sq)
            p3                  =   torch.exp(-d3.sum(dim=2)/sigma_sq)
            
            l0                  =   torch.log((p1+p3)/(p0+p2+1e-8))
            l1                  =   torch.log((p2+p3)/(p0+p1+1e-8))
            
            l0                  =   l0.unsqueeze(2)
            l1                  =   l1.unsqueeze(2)
            
            llr                 =   torch.cat([l1,l0],dim=2)

            llr_values          =   llr.reshape(batch_size,-1)
        
        return llr_values
    
    def __call__(self,cwd_tensor):
        
        modulated_tensor    =   self.modulate(cwd_tensor)
        noisy_tensor,sigma  =   self.add_noise(modulated_tensor)
        llr_values          =   self.demodulate(noisy_tensor,sigma)
        
        return llr_values


def find_correct(out_dec,messages):
    
    out_dec     =   out_dec.detach()
    messages    =   messages.detach()
    
    out_dec     =   out_dec.round()
    
    diff        =   out_dec - messages
    diff_abs    =   1.0 - diff.abs()
    
    sum_diff    =   diff_abs.sum()
    
    return sum_diff


def get_args(): 
    
    parser      =   argparse.ArgumentParser()
    
    parser.add_argument('--exp-dir',type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    parser.add_argument('--n-iteration',type = int ,default = 5)
    
    parser.add_argument('--test-snr-low',type=float,default =0.0)
    parser.add_argument('--test-snr-high',type=float,default=9.0)
    parser.add_argument('--test-snr-step',type=float,default=0.5)
    
    parser.add_argument('--numblocks',type=int,default =200)
    parser.add_argument('--batch-size',type=int,default=40)
    
    parser.add_argument('--test-numblocks',type=int,default =200)
    parser.add_argument('--test-batch-size',type=int,default=40)
    
    parser.add_argument('--check-every',type=int,default=20)
    
    parser.add_argument('--mod', type=str, default='bpsk')
    parser.add_argument('--n', type=int, default=127)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--ebno-esno', type=str, default='ebno')
    parser.add_argument('--systematic-encoder', type=bool, default=True)
    
#    parser.add_argument('--power',type=float,default =1.0)
    
    parser.add_argument('--training-snr-point',type=float,default=6.0, help='All SNR in EsN0.')
    parser.add_argument('--training-snr-low',type=float,default = 1.0)
    parser.add_argument('--training-snr-high',type=float,default= 7.0)
    parser.add_argument('--training-type',type=str,default='range')  #'range','fix')
    parser.add_argument('--arch-disc',type=str,default='almost_final_trial')
    parser.add_argument('--test-in-range',type=bool,default = True)
    
    args = parser.parse_args()
    
    return args

class decoder(nn.Module):
    
    def __init__(self,m,n,H,iterations):
        super(decoder,self).__init__()

        self.m          =   m
        self.n          =   n
        self.H          =   H.astype(int)
        
        self.tanner_graph   =   graph_structure(self.m,self.n,self.H)
#        print(self.tanner_graph)
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
#        print(self.total_edges)
#        print(self.total_weights)
        
        
        for iter_idx in range(self.iterations):
            
            a = nn.Parameter(torch.ones(self.total_weights).double().to(device),requires_grad = True)
            name = 'iter_{}_edge_params'.format(iter_idx)
            self.w_param[name]  = a
            a = nn.Parameter(torch.ones(self.n).double().to(device),requires_grad = True)
            name = 'iter_{}_llr_params'.format(iter_idx)
            self.llr_param[name] = a
                
        
        a = nn.Parameter(torch.ones(self.total_edges).double().to(device),requires_grad = True)
        name = 'final_edge_params'
        self.w_param[name]   = a
        a = nn.Parameter(torch.ones(self.n).double().to(device),requires_grad = True)
        name = 'final_llr_params'
        self.llr_param[name]  = a
        
    
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
            
            if i1!=i2:
                message_to_edge_i   =   weighted_check_var_sets[:,i1:i2]
                message_to_edge_i   =   message_to_edge_i.sum(dim=1)
                message_to_edge_i   =   message_to_edge_i.unsqueeze(1)
                
                message_to_edge_i   =   message_to_edge_i + weighted_llr[:,llr_no].unsqueeze(1)
                weighted_check_var.append(message_to_edge_i)
            else:
                message_to_edge_i   =   weighted_llr[:,llr_no].unsqueeze(1)
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
            if i1!=i2:
                message_to_edge_i   =   var_to_check_prod_sets[:,i1:i2]
                message_to_edge_i   =   message_to_edge_i.prod(dim=1)
                message_to_edge_i   =   message_to_edge_i.unsqueeze(1)
                msg_sets.append(message_to_edge_i)
            else:
                message_to_edge_i   =   torch.ones(batch_size,1,requires_grad=True).double().to(device)
                msg_sets.append(message_to_edge_i)
        
        msg_sets      =   torch.cat(msg_sets,dim=1)
#        print(msg_sets.device)
        check_to_var_msg    =   2*torch.atanh(0.9995*msg_sets)   
        
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
            if i1!=i2:
                confidence_node_i   =   weighted_check_var_sets[:,i1:i2]
                confidence_node_i   =   confidence_node_i.sum(dim=1)
                confidence_node_i   =   confidence_node_i.unsqueeze(1)
                confidence_node_i   =   confidence_node_i + weighted_llr[:,llr_no].unsqueeze(1)
                final_output.append(confidence_node_i)
            else:
                confidence_node_i   =   weighted_llr[:,llr_no].unsqueeze(1)
                final_output.append(confidence_node_i)
        
        final_output      =   torch.cat(final_output,dim=1)
        
#        final_output        =   self.tanh(0.5*final_output)
        
        return final_output
    
    def forward(self,llr):
        
        # Zero belief from check nodes in the begging.
        check_to_var_msg        =   torch.zeros(llr.shape[0],self.total_edges).double().to(device)
        
        for iter_idx in range(self.iterations):
            var_to_check_msg   =   self.var_to_check_calc(check_to_var_msg,llr,iter_idx)

            check_to_var_msg    =   self.check_to_var_calc(var_to_check_msg,iter_idx)
        
        final_output        =   self.final_output_calc(check_to_var_msg,llr)
        
        output          =   final_output    # [:,:self.n-self.m]
        
        output          =   self.sigmoid(output)
        
        return output #, stack_of_outputs



def train(epoch_number):
    
    num_batches        =   int(args.numblocks/args.batch_size)
    
    dec_obj.train()
    
    train_loss          =   0.0
    correct             =   0.0
    
    for idx in range(num_batches): 
        
        if args.training_type == 'range':
            snr_db      =   [args.training_snr_low,args.training_snr_high]
        else:
            snr_db      =   [args.training_snr_point]

#       def __init__(self,noise_type='awgn',choice='bpsk',snr_db=0.0,rate=1,ebno_esno='ebno',device='cuda'):
        channel     =   channel_model('awgn', args.mod, snr_db, rate, args.ebno_esno, device)
            
        optimizer_dec.zero_grad()
        
        messages        =   np.random.randint(0,2,(args.batch_size,k))
        
        all_codewords   =   enc_obj(messages)
        llr_values      =   channel(all_codewords)
#        out_dec, stack_of_outputs =   dec_obj(llr_values)
        out_dec =   dec_obj(llr_values)
        
#        print(out_dec)
#        print(all_codewords)
        
        if args.systematic_encoder:
            loss            =   bce_loss(out_dec[:,:args.k],all_codewords[:,:args.k])
        else:
            loss            =   bce_loss(out_dec,all_codewords)
        
#        for i in range(0,stack_of_outputs.shape[0]-1,1):
#            loss        =   loss + (0.9**(stack_of_outputs.shape[0]-i))*bce_loss(stack_of_outputs[i],all_codewords)
        
        temp_loss       =   loss.item()
        
        if args.systematic_encoder:
            temp_correct        =   find_correct(out_dec[:,:args.k], all_codewords[:,:args.k])
        else:
            temp_correct        =   find_correct(out_dec, all_codewords)
        
        loss.backward()
        optimizer_dec.step()
            
        del loss
        train_loss                     += temp_loss  # loss.item()
        correct                        +=temp_correct
        
    
    if args.systematic_encoder:
        train_acc            =   float(correct)/(args.k*num_batches*args.test_batch_size)
    else:
        train_acc            =   float(correct)/(args.n*num_batches*args.test_batch_size)
    
    print('train_acc = ',train_acc)
    train_ber       =   1.0-train_acc
    
    str_print       =   "\nEpoch Number {:}: \t Train Loss = {:}, \t\t Train BER = {:}".format(epoch_number,train_loss,train_ber)
    
    print(str_print)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_loss, train_ber



def test(epoch_number, snr_db = None):
    
    dec_obj.eval()
    
    if snr_db is None:
        snr_db          =   [args.training_snr_point] 
    else:
        snr_db          =   [snr_db]
    
    channel         =   channel_model('awgn', args.mod, snr_db, rate, args.ebno_esno, device)
    print(channel)
    test_loss           =   0.0
    correct             =   0.0
    
    num_batches         =   int(args.test_numblocks/args.test_batch_size)
    
    with torch.no_grad():
            
        for idx in range(num_batches):     
            messages            =   np.random.randint(0,2,(args.test_batch_size,k))
            all_codewords       =   enc_obj(messages)
            llr_values          =   channel(all_codewords)
#            out_dec, __         =   dec_obj(llr_values)
            out_dec         =   dec_obj(llr_values)
                
            if args.systematic_encoder:
                loss            =   bce_loss(out_dec[:,:args.k],all_codewords[:,:args.k])
            else:
                loss            =   bce_loss(out_dec,all_codewords)
                
            test_loss          +=   loss.item()
            
            if args.systematic_encoder:
                temp_correct        =   find_correct(out_dec[:,:args.k], all_codewords[:,:args.k])
            else:
                temp_correct        =   find_correct(out_dec, all_codewords)
            
            correct            +=   temp_correct
            
#    print('value of correct is : ',correct)
#    print('divisor is : ',args.k1 * args.batch_size * num_batches)
            
    if args.systematic_encoder:
        test_acc            =   float(correct)/(args.k*num_batches*args.test_batch_size)
    else:
        test_acc            =   float(correct)/(args.n*num_batches*args.test_batch_size)
        
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
    
    for snr_db in np.arange(l, h, step):
        
        __, ber_val     =   test(epoch,snr_db)
        
        test_ber_vals.append(ber_val)
        test_snr_vals.append(snr_db)
    
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
    
    args            =   get_args()
    
    args.test_numblocks =   args.numblocks
    args.test_batch_size    =   args.batch_size
    
#    assert args.numblocks       ==  args.test_numblocks
#    assert args.test_batch_size ==  args.batch_size
        
    profiler        =   profiler_class(args)
    
    n               =   args.n
    k               =   args.k
    m               =   n-k
    
    generator_mat, parity_matrix    =   get_bch_G_H_mat(n,k,args.systematic_encoder)
#    print(parity_matrix)
        
#    H = np.array([[1,1,1,1,0,0],
#                  [0,1,1,0,1,0],
#                  [1,0,1,0,0,1]])
    
#    H = np.array([[1,1,1,0,1,0,0],
#                  [1,1,0,1,0,1,0],
#                  [1,0,1,1,0,0,1]]  )
    
#    G = np.array([[1,0,0,1,0,1],
#                  [0,1,0,1,1,0],
#                  [0,0,1,1,1,1]])
    
#    G   =   np.array([[1,0,0,0,1,1,1],
#                      [0,1,0,0,1,1,0],
#                      [0,0,1,0,1,0,1],
#                      [0,0,0,1,0,1,1]])

#    generator_mat, parity_matrix        =   G, H
    
    enc_obj         =   encoder_class(n,k,generator_mat)
    dec_obj         =   decoder(m,n,parity_matrix,args.n_iteration).double().to(device)
    
    rate            =   k/n
    
    bce_loss        =   nn.BCELoss()
    
    lr              =   args.lr
    check_every     =   args.check_every
    curr_checkpoint =   0
    
    training_ber_curve  =   []
    testing_ber_curve   =   []
    
#    print(list(dec_obj.parameters()))
    
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
