# BP Decoder

As a part of my thesis work at IIIT Hyderabad under Dr. Lalitha Vadlamani, I have implemented a code for the Belief Propagation decoder for Linear block codes.
As it is known that the decoder performance of a BP (Belief Propagation) decoder depends on the number of small cycles in the bipartite tanner graph, there have been explorations (references below) where a decoder utilising weighted edges in a tanner graph is able to perform better than plain BP decoder. My code is a implementation such a deocder similar to that in [1].

File descriptions:

BCH Code Library : This file generates the generator and parity check matrices for BCH codes. The generated matrices can be used for belief propogation decoding of BCH codes.

Buid graph       : This file generates a tanner graph for the parity check matrix of a linear block code. This generated graph is used in the main file for doing decoding.

Channel Library  : This file is the channel implementation of AWGN channel. The channel takes in the codeword tensors, does its modulation, adds noise to it and then gives out the LLR values for the reciever.

Profiler module  : This is a profiler implementation which is used in the main file for the recording of BER metrics when the decoding is done with various experiment settings.


References:

[1] E. Nachmani, Y. Be’ery, and D. Burshtein. Learning to decode linear codes using deep learning. In 2016 54th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 341–346, 2016.

[2] E. Nachmani, E. Marciano, D. Burshtein, and Y. Be’ery. RNN decoding of linear block codes, 2017.

