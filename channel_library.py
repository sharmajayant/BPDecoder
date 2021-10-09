from __future__ import division
import numpy as np

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

