# BP Decoder

This code is an implementation of the Belief Propagation decoder for a Linear block code. 

File descriptions:

BCH Code Library : This file generates the generator and parity check matrices for BCH codes. The generated matrices can be used for belief propogation decoding of BCH codes.

Buid graph       : This file generates a tanner graph for the parity check matrix of a linear block code. This generated graph is used in the main file for doing decoding.

Channel Library  : This file is the channel implementation of AWGN channel. The channel takes in the codeword tensors, does its modulation, adds noise to it and then gives out the LLR values for the reciever.

Profiler module  : This is a profiler implementation which is used in the main file for the recording of BER metrics when the decoding is done with various experiment settings.

