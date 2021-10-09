from __future__ import division
import argparse
from time import strftime
from time import time
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib
#import matplotlib.animation as animation
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class profiler_class():
    
    def __init__(self,args):
        self.time_str       =   strftime('%d_%b_%Hh_%Mm_%Ss')
        self.end_time_str   =   ''
        self.begin_time     =   time()
        self.init_exp_dir(args)
        self.dict_args      =   dict(vars(args))
        self.print_args()
        self.create_exp_dir()
        self.save_args()
        self.copy_experiment_file()
    
    def init_exp_dir(self,args):
        
        time_dir        =   self.begin_time
        arch_dir        =   args.arch_disc
        
        try:
            enc_dir         =   args.enc_directions + '_' + args.enc_rnn + '_' +str(args.enc_rnn_layers) + '_Hidden_' + str(args.enc_hidden_size) + '_Filters_' + str(args.enc_cnn_filters)
        except:
            enc_dir     =   ''
        
        try:
            dec_dir         =   args.dec_directions + '_' + args.dec_rnn + '_' +str(args.dec_rnn_layers) + '_Hidden_' + str(args.dec_hidden_size) + '_Filters_' + str(args.dec_cnn_filters)
        except:
            dec_dir     =   ''
        
        variant_dir     =   'Enc_' + enc_dir + '_Dec_' + dec_dir
        exp_dir         =   os.path.join(arch_dir, variant_dir, self.time_str)
        self.exp_dir    =   exp_dir
        args.exp_dir    =   exp_dir
        
    def print_args(self):
        for key, value in sorted(self.dict_args.items()):
            print('{} = {}'.format(key,value))
    
    def create_exp_dir(self):
        os.makedirs(self.exp_dir)
    
    def copy_experiment_file(self):
        
        curr_file_str   =   sys.argv[0]
        curr_file       =   open(curr_file_str)
        file_contents   =   curr_file.read()
        curr_file.close()
        
        fname           =   os.path.join(self.exp_dir,'experiment_file.py')
        experiment_file =   open(fname,'w+')
        experiment_file.write(file_contents)
        experiment_file.close()
        
    def save_args(self):
        fname           =   os.path.join(self.exp_dir,'arguments_used.txt')
        arguments_file  =   open(fname,'w+')
        
        for key, value in sorted(self.dict_args.items()):
            arguments_file.write('{} = {}\n'.format(key,value))
        arguments_file.close()
    
    def logs(self,enc_obj,dec_obj,test_in_range,checkpoint=-1,modulator_obj=None,demodulator_obj=None):
        
        if checkpoint != -1:
            self.chk_dir= os.path.join(self.exp_dir,'checkpoint {}'.format(checkpoint))
            os.makedirs(self.chk_dir)
        else:
            self.chk_dir= self.exp_dir
        
        self.exec_time()
        self.save_models(enc_obj,dec_obj,modulator_obj,demodulator_obj)
        if self.dict_args['test_in_range'] == True:
            self.range_test(test_in_range)
        
    
    def final_logs(self,training_ber_curve, testing_ber_curve):
        self.save_learning_curve(training_ber_curve,testing_ber_curve)
        self.save_learning_graphs(training_ber_curve,testing_ber_curve)
    
    
    def range_test(self,test_in_range):
        
        test_snr_vals, test_ber_vals = test_in_range()
        write_these                 =   {'SNR_value ':test_snr_vals}
        write_these['BER_value ']   =   test_ber_vals
        filename                    =   os.path.join(self.chk_dir,'ber_values.csv')
        rng_test_file               =   open(filename,'w')
        df                          =   pd.DataFrame(data=write_these,dtype=np.float32)
        df.to_csv(rng_test_file)
        rng_test_file.close()
        
        marker_style                =   '-'
        plot_title                  =   'SNR vs BER values.'
        x_label                     =   'SNR_value'
        y_label                     =   'BER_value'
        file_name                   =   os.path.join(self.chk_dir,'semilog_range_test.jpg')
        plt.semilogy(test_snr_vals, test_ber_vals, marker_style, label='Neural net BER_values')
        plt.grid(True,which='both',ls=':',color='0.0')
        plt.legend(loc=1, prop={'size': 6})
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.savefig(file_name, format='jpg', bbox_inches='tight', dpi=300)
        plt.close()
        
    
    def save_models(self,enc_obj,dec_obj,modulator_obj=None,demodulator_obj=None):
        model_dir       =   os.path.join(self.chk_dir, 'models')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        
        if enc_obj is not None:
            filename        =   os.path.join(model_dir, 'encoder.pth')
            torch.save(enc_obj, filename)
        
        if dec_obj is not None:
            filename        =   os.path.join(model_dir, 'decoder.pth')
            torch.save(dec_obj, filename)

        if modulator_obj is not None:
            filename        =   os.path.join(model_dir, 'modulator.pth')
            torch.save(modulator_obj, filename)
            
        if demodulator_obj is not None:
            filename        =   os.path.join(model_dir, 'demodulator.pth')
            torch.save(demodulator_obj, filename)
        
    
    def save_learning_graphs(self,training_ber_curve,testing_ber_curve):
        
        epochs                  =   np.arange(1,len(training_ber_curve)+1)
        
        marker_style            =   '-'
        legend_strings          =   ['training_ber_curve','testing_ber_curve']
        x_label                 =   'Epochs'
        y_label                 =   'BER_values'
        plot_title              =   'Learning Curves.'
        
        file_name               =   os.path.join(self.exp_dir,'learning_curves.png')
        plt.semilogy(epochs, training_ber_curve, marker_style, label=legend_strings[0])
        plt.semilogy(epochs, testing_ber_curve, marker_style, label=legend_strings[1])
        plt.grid(True,which='both',ls=':',color='0.0')
        plt.legend(loc=1, prop={'size': 6})
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.savefig(file_name, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        
#        file_name               =   os.path.join(self.exp_dir,'semilog_learning_curves.eps')
#        plt.semilogy(epochs, training_ber_curve, marker_style, label=legend_strings[0])
#        plt.semilogy(epochs, testing_ber_curve, marker_style, label=legend_strings[1])
#        plt.grid(True,which='both',ls=':',color='0.0')
#        plt.legend(loc=1, prop={'size': 6})
#        plt.title(plot_title)
#        plt.xlabel(x_label)
#        plt.ylabel(y_label)
#        
#        plt.savefig(file_name, format='eps', bbox_inches='tight', dpi=300)
#        plt.close()
    
    
    def save_learning_curve(self,training_ber_curve,testing_ber_curve):
        
        epochs                      =   np.arange(1,len(training_ber_curve)+1)
        write_these                 =   {'Epoch ':epochs}
        write_these['BER_value ']   =   training_ber_curve
        filename                    =   os.path.join(self.exp_dir,'training_ber_curve.csv')
        learning_curve_file         =   open(filename,'w')
        df                          =   pd.DataFrame(data=write_these,dtype=np.float32)
        df.to_csv(learning_curve_file)
        learning_curve_file.close()
        
        epochs                      =   np.arange(1,len(testing_ber_curve)+1)
        write_these                 =   {'Epoch ':epochs}
        write_these['BER_value ']   =   testing_ber_curve
        filename                    =   os.path.join(self.exp_dir,'testing_ber_curve.csv')
        learning_curve_file         =   open(filename,'w')
        df                          =   pd.DataFrame(data=write_these,dtype=np.float32)
        df.to_csv(learning_curve_file)
        learning_curve_file.close()
    
    
    
    def exec_time(self):
        
        self.begin_time_str =   'Program_started_at {}.'.format(self.time_str)
        self.end_time_str   =   strftime('Program_ended_at   %d_%b_%Hh_%Mm_%Ss.')
        print(self.begin_time_str)
        print(self.end_time_str)
        
        self.end_time       =   time()
        total_time          =   self.end_time - self.begin_time
        hours               =   np.floor(total_time/3600)
        minutes             =   np.floor((total_time-hours*3600)/60)
        seconds             =   np.ceil(total_time-hours*3600-minutes*60)
        
        string_to_write     =   '{} Hours, {} Minutes, {} Seconds.'.format(hours,minutes,seconds)
        
        filename            =   os.path.join(self.chk_dir,'Execution_time.txt')
        file_obj            =   open(filename,'w+')
        file_obj.write(self.begin_time_str)
        file_obj.write('\nProgram_executed_for {}.'.format(string_to_write))
        file_obj.write('\n{}'.format(self.end_time_str))
        file_obj.close()
    
    

