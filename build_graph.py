import numpy as np
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
        
        self.marginalize_ind = np.array(self.marginalize_ind)
        self.marginalize_ind = self.marginalize_ind.cumsum()
        
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
    
    def __str__(self):
        
        string_to_return = ""
#        
#        string_to_return = string_to_return + str(self.v_sum_indices)
#        
#        for index in range(self.total_edges):
#            v_num            =   self.edge_set[index][1]
#            string_to_return =  string_to_return + '\n'
#            string_to_return =  string_to_return + str(index) + ' Σ '+ str(self.v_c_edges[index])
#            string_to_return =  string_to_return + '\n' + str(self.dvi[v_num]-1)
#        
#        
#        string_to_return = string_to_return + '\n'+ str(self.c_prod_indices)
#        
#        for index in range(self.total_edges):
#            c_num   =   self.edge_set[index][0]
#            string_to_return = string_to_return + '\n'
#            string_to_return = string_to_return + str(index) + ' ᴨ ' + str(self.c_v_edges[index])
#            string_to_return = string_to_return + '\n' + str(self.dci[c_num]-1)
#        
#        string_to_return    =   string_to_return + '\n' + str(self.ind_final_vsum)
#        string_to_return    =   string_to_return + '\n' + str(self.marginalize_ind)
#        
#        for index in range(self.n):
#            v_num            =   self.edge_set[index][1]
#            string_to_return =  string_to_return + '\n'
#            string_to_return =  string_to_return + str(index) + ' Σ '+ str(self.v_c_edges[index])
#            string_to_return =  string_to_return + '\n' + str(self.dvi[v_num]-1)
        
        string_to_return = string_to_return + str(self.v_sum_indices) + '\n'
        
        for index in range(self.total_edges):
            i1  =   self.ind_vsum[index]
            i2  =   self.ind_vsum[index+1]
            to_sum = self.v_sum_indices[i1:i2]
            llr_no  =   self.edge_set[index][1]

            string_to_return = string_to_return + '\n e{} ='.format(index)  + ' Σ '+ str(to_sum) + ' llr{}'.format(llr_no)
        
        string_to_return = string_to_return + '\n' + str(self.c_prod_indices) + '\n'
        
        for index in range(self.total_edges):
            i1  =   self.ind_cprod[index]
            i2  =   self.ind_cprod[index+1]
            to_prod = self.c_prod_indices[i1:i2]
            string_to_return = string_to_return + '\n e{} ='.format(index)  + ' ᴨ ' + str(to_prod)
            if i1 ==i2:
                string_to_return = string_to_return + '|1|'
                
        string_to_return = string_to_return + '\n' + str(self.ind_final_vsum) + '\n'
        
        for index in range(self.n):
            i1  =   self.marginalize_ind[index]
            i2  =   self.marginalize_ind[index+1]
            to_sum = self.ind_final_vsum[i1:i2]
            
            string_to_return = string_to_return + '\n e{} ='.format(index)  + ' Σ ' + str(to_sum) + ' llr{}'.format(index)
        
            
        return string_to_return
        
