from __future__ import division
import numpy as np

#class gf2():
#    def __init__(self,x):
#        assert  (x==0) or (x==1)
#        self.x = x
#        
#    def __add__(self,second_value):
#        sum_val     =   (self.x + second_value.x)%2
#        ret_obj     =   gf2(sum_val)
#        return ret_obj
#        
#    def __str__(self):
#        return "GF2 value = {}".format(self.x)
#                
#    def __repr__(self):
#        return "GF2 value = {}".format(self.x)
#    
#    def __sub__(self,second_value):
#        return self.__add__(second_value)
#    
#    def __mul__(self,second_value):
#        
#        if (self.x == 1) and (second_value.x ==1):
#            output  =   gf2(1)
#        else:
#            output  =   gf2(0)
#        return output

dict_polynomial = { 15  + 7   : [1,1,1,  0,1,0,  0,0,1],                             #721,
                    31  + 16  : [    1,  0,0,0,  1,1,1,  1,1,0,  1,0,1,  1,1,1],     #107657,
                    31  + 21  : [  1,1,  1,0,1,  1,0,1,  0,0,1],                     #3551,
                    63  + 51  : [    1,  0,1,0,  1,0,0,  1,1,1,  0,0,1],             #12471,
                    127 + 64  : [    1,  0,1,0,  0,0,0,  1,1,0,  1,0,1,  0,1,1,  1,0,0,  0,0,0,  0,1,0,  1,0,1,  1,0,1,  1,1,1,  0,0,0,  1,1,1,  1,1,1,  0,1,1,  0,0,1,  0,0,0,  0,0,0,  0,0,0,  1,0,0,  1,0,1],      #12065 34025 57077 31000 45
                    127 + 113 : [1,0,0,  0,0,1,  1,0,1,  1,1,0,  1,1,1],             #41567,
                    255 + 239 : [  1,0,  1,1,0,  1,1,1,  1,0,1,  1,0,0,  0,1,1]      #267543
                   }
        
        # ref costello
        
t_table= { 15  + 7   : 2,    #721,
           31  + 16  : 3,    #107657,
           31  + 21  : 2,    #3551,
           63  + 51  : 2,    #12471,
           127 + 64  : 10,
           127 + 113 : 2,    #41567,
           255 + 239 : 2    #267543
          }


class alpha_element():
    
    @staticmethod
    def primitive_polynomial(exp_m):
        lookup  =   {}
        lookup[2]   =   (0,1,2)
        lookup[3]   =   (0,1,3)
        lookup[4]   =   (0,1,4)
        lookup[5]   =   (0,2,5)
        lookup[6]   =   (0,1,6)
        lookup[7]   =   (0,3,7)
        lookup[8]   =   (0,2,3,4,8)
        poly_tuple = lookup[exp_m]
        return poly_tuple
    
    @staticmethod
    def create_table(exp_m):
        
        alpha_vectors   =   {}
        prim_poly       =   alpha_element.primitive_polynomial(exp_m)
        assert prim_poly[-1] == exp_m
        prim_poly       =   prim_poly[:-1]
        max_power       =   (2**exp_m)-1
        eye_matrix      =   np.eye(exp_m)
        
        all_repr_sets   =   {}
        
        for i in range(exp_m):
            alpha_vectors[i] = eye_matrix[i]  
            all_repr_sets[i] = {i}
        
        count = 0
        for i in range(exp_m,max_power,1):
            temp    =   alpha_vectors[0]*0
        
            temp_list   =   []
            for j in prim_poly:
                idx     =   j + count
                temp    =   (temp + alpha_vectors[idx])%2
                temp_list.append(idx)
            
            count = count + 1
            alpha_vectors[i]    =   temp
            all_repr_sets[i]    =   set(temp_list)
            
        
        alpha_element.max_power     =   max_power
        alpha_element.alpha_vectors =   alpha_vectors
        alpha_element.exp_m         =   exp_m
        alpha_element.prim_poly     =   prim_poly
        alpha_element.all_repr_sets =   all_repr_sets
        print('alpha Table is updated with m = {}'.format(exp_m))
    
    def __init__(self,power_value=1):
        self.power_value    =   power_value % alpha_element.max_power
        self.repr_tuple     =   alpha_element.all_repr_sets[self.power_value]
    
    def __call__(self,power_value):
        return alpha_element(power_value)
    
    def __pow__(self,value):
        value   =   self.power_value*value
        return  alpha_element(value)
    
    def __mul__(self,other_element):
        power_value         =   self.power_value + other_element.power_value
        return alpha_element(power_value)
        
    def __add__(self,other_element):
        
        raise NotImplemented
        
    def vector_repr(self):
        vector_to_return    =   alpha_element.alpha_vectors[self.power_value]
        vector_to_return    =   vector_to_return.reshape(alpha_element.exp_m,1)
        return vector_to_return
    
    def __str__(self):
        
        string_to_return    =   'a^{}'.format(self.power_value)
        vector_repr         =   self.vector_repr()
        string_to_return    =   string_to_return + '\n' + str(vector_repr)
    
        return string_to_return
        
    def __repr__(self):
        
        string_to_return    =   'a^{}'.format(self.power_value)
        vector_repr         =   self.vector_repr()
        string_to_return    =   string_to_return + '\n' + str(vector_repr)
    
        return string_to_return
    
    
    def print_table(self,):
        
        for key,value in alpha_element.alpha_vectors.items():
            print('a^{}\n'.format(key))
            print(value)
            


def get_bch_G_H_mat(n,k,systematic=True):
    
    if n%2 ==0:
        exp_m   = int(np.log2(n))
        val                 =   k + n -1
    else:
        exp_m   = int(np.log2(n+1))
        val                 =   k+n
    
    
    if systematic:
        generator_mat, parity_matrix =  get_bch_G_H_mat_sys(n,k,exp_m)
    else:
        print('In the non systematic convention MSB is Last with highest degree coeff as last value in msg.')

        alpha_element.create_table(exp_m)
        a1  =   alpha_element(1)
    #    a1.print_table()
        
        m   =   n-k
        rows    =   t_table[val]     #int((d-1)/2)
        cols    =   n-1 if n%2 == 0 else n
        
        parity_matrix   =   np.array([]).reshape(0,cols)
        for i in range(rows):
            current_row =   np.array([]).reshape(exp_m,0)
            odd_value   =   2*i+1
            for j in range(cols):
                power_value =   j*odd_value
                a_power     =   a1**power_value
                vector      =   a_power.vector_repr()
                current_row =   np.append(current_row,vector,axis=1)
            
            parity_matrix   =   np.append(parity_matrix,current_row,axis=0)
        
        if n%2  ==0:
            col_zeros           =   np.zeros((exp_m*rows,1))
            parity_matrix       =   np.append(parity_matrix,col_zeros,axis=1)
            row_ones            =   np.ones((1,n))
            parity_matrix       =   np.append(parity_matrix,row_ones,axis=0)
#            print('Returning the polynomial for BCH({},{}).\nAppend additional parity for the desired n.'.format(n-1,k))
#        else :
#            print('Returning the polynomial for BCH({},{}).\nNot required to append additional parity for the desired n.'.format(n,k))
        
        polynomial      =   np.array(dict_polynomial[val])
        polynomial      =   np.array(polynomial[-1::-1])      # For the order of bits will be reverse in parity_matrix.
        polynomial      =   polynomial.astype(float)
        
        parity_matrix    =   parity_matrix.astype(float)

        if n%2 ==0:
            generator_mat   =   np.zeros((0,n-1))
        else:
            generator_mat   =   np.zeros((0,n))
            
        for i in range(k):
            shift_mat       =   np.zeros(i)
            current_row     =   np.append(shift_mat,polynomial,axis=0)
            if n%2==0:
                current_row     =   np.append(current_row,np.zeros(n-current_row.shape[0]-1),axis=0).reshape(1,-1)
            else: 
                current_row     =   np.append(current_row,np.zeros(n-current_row.shape[0]),axis=0).reshape(1,-1)
                
            generator_mat   =   np.append(generator_mat,current_row,axis=0)
        
        if n%2==0:
            g_sum_col       =   generator_mat.sum(axis=1) % 2
            g_sum_col       =   g_sum_col.reshape(-1,1)
            generator_mat   =   np.append(generator_mat, g_sum_col ,axis =1)
    
    if validate_codebook(generator_mat,parity_matrix) == False:
        print('Wrong polynomial parity_matrix pair generated. Exiting the program.')
        raise Exception('Bad pair of encoder and decoder.')
    
    
    
    return generator_mat, parity_matrix
    

def validate_codebook(generator_mat, parity_matrix):
    
    flag = True
    
#    print(parity_matrix)
    
#    shape       =   parity_matrix.shape
#    m           =   shape[0]
#    n           =   shape[1]
#    k           =   n-m
    
    HcTranspose =   np.matmul(parity_matrix,generator_mat.T)
    HcTranspose =   HcTranspose.sum() %2
        
    if HcTranspose != 0:
        flag = False
    
    print('Encoder-decoder pair is verified as a valid pair : {}'.format(flag))
    
    return flag


def get_bch_G_H_mat_sys(n,k,exp_m):
    
    print('In the systematic convention MSB is first with highest degree coeff as first value in msg.')
    
    m   =   n-k
    
    if n%2  ==0:
        val                 =   k + n -1
#        print('Returning the polynomial for BCH({},{}).\nAppend additional parity for the desired n.'.format(n-1,k))
    else :
        val                 =   k+n
#        print('Returning the polynomial for BCH({},{}).\nNot required to append additional parity for the desired n.'.format(n,k))
    
    polynomial      =   np.array(dict_polynomial[val])
#    polynomial      =   np.array(polynomial[-1::-1])      # For the order of bits will be reverse in parity_matrix.
    polynomial      =   polynomial.astype(float)
    
    msgs        =   np.eye(k)
    if n%2==0:
        shift       =   np.zeros((k,m-1))
        remainder_stack =   np.zeros((0,m-1))
    else:
        shift       =   np.zeros((k,m))
        remainder_stack =   np.zeros((0,m))
    
    # Shift the msgs.
    shifted_msgs    =   np.append(msgs,shift,axis=1)
    
    
    for i in range(k):
        quotient,remainder  =   np.polydiv(shifted_msgs[i],polynomial)
        remainder           =   remainder.reshape(1,-1)  %2
        if n% 2 ==0:
            append_zeros        =   np.zeros((1,m-remainder.shape[1]-1))
        else:
            append_zeros        =   np.zeros((1,m-remainder.shape[1]))
            
        remainder           =   np.append(append_zeros,remainder,axis=1)
        remainder_stack     =   np.append(remainder_stack,remainder,axis=0) 
    
    codeword            =   np.append(msgs,remainder_stack,axis=1)
    
    generator_mat       =   codeword
    
    parity_matrix       =   codeword[:,k:]
    parity_matrix       =   parity_matrix.T
    
    if n%2 ==0:
        sum_column          =   generator_mat.sum(axis=1).reshape(-1,1)
        sum_column          =   sum_column %2
        temp                =   np.append(sum_column,generator_mat[:,k:],axis=1)
#        print(temp.shape)
#        print(generator_mat.shape)
#        print(m)
        generator_mat       =   np.append(np.eye(k),temp,axis=1)
        
    
    parity_matrix       =   np.append(generator_mat[:,k:].T,np.eye(m),axis=1)
    
    
#    print('generator mat is : \n{}'.format(generator_mat))
#    print('parity_matrix is  : \n{}'.format(parity_matrix))
    
#    print('Shapes are : {}, and {}'.format(generator_mat.shape,parity_matrix.shape))
    
#    print(parity_matrix[-1])
#    print(generator_mat[:,-1])
#    res                 =   np.matmul(generator_mat,parity_matrix.T) %2 
#    if res.sum()==0:
#        print('Encoder-decoder pair is verified as a valid pair : {}'.format('True'))
#    else:
#        print('Codebook generated is invalid.')
    
    return generator_mat, parity_matrix
    

#def validate(polynomial, parity_matrix):
#    
#    flag = True
#    
##    print(parity_matrix)
##    print(polynomial)
#    
#    shape       =   parity_matrix.shape
#    m           =   shape[0]
#    n           =   shape[1]
#    k           =   n-m
#    

#    
#    append_parity   =   1 if n%2 ==0 else 0
#    
#    msgs        =   np.eye(k)
#    
#    for i in range(k):
#        codeword    =   np.convolve(msgs[i],polynomial) %2 
#        
#        if append_parity:
#            parity      =   np.array([codeword.sum() %2])
#            codeword    =   np.append(codeword,parity,axis = 0)
#        
#        HcTranspose =   np.matmul(parity_matrix,codeword)
##        print(HcTranspose)
#        HcTranspose =   HcTranspose.sum() %2
#        
#        if HcTranspose != 0:
#            flag = False
#            break
#    
#    print('Encoder-decoder pair is verified as a valid pair : {}'.format(flag))
#    
#    return flag
#    




if __name__ == "__main__":

#    exp_m   =   2
#    alpha_element.create_table(exp_m)
#    a   =   alpha_element(0)
#    b   =   alpha_element(1)
#    print(a)
#    print(b)
#    print(c)
    
    
    n       =   15
    k       =   7
    systematic  =   True
    
#    alpha_element.create_table(exp_m)
    
    generator_mat, bch_matrix =   get_bch_G_H_mat(n,k,systematic)
    generator_mat   =   generator_mat.astype(int)
    bch_matrix      =   bch_matrix.astype(int)
    print('Generator matrix is : \n{}'.format(generator_mat))
    print('Parity check matrix is \n{}'.format(bch_matrix))
    
    
#    __,__ = get_bch_G_H_mat_sys(n,k,systematic)
    

