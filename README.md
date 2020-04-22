# fasterai
FasterAI: A repository for making smaller and faster models with the FastAI library.


Currently implemented: 

- Network Sparsifying: make the network sparse (i.e replace some weight by zero)

- Network Pruning: actually remove the sparse weights to take computationnally take advantage of it.

- Batch Normalization Folding: remove the batch normalization layer to reduce the number of parameters and the inference time without changing the performance
