# fasterai
FasterAI: A repository for making smaller and faster models with the FastAI library.


Currently implemented: 

- **Network Sparsifying**: make the network sparse (i.e replace some weight by zero). [Blog post here](https://nathanhubens.github.io/posts/deep%20learning/2020/05/22/pruning.html)

- **Network Pruning**: actually remove the sparse weights to take computationnally take advantage of it. [Blog post here](https://nathanhubens.github.io/posts/deep%20learning/2020/05/22/pruning.html)

- **Batch Normalization Folding**: remove the batch normalization layer to reduce the number of parameters and the inference time without changing the performance [Blog post here](https://nathanhubens.github.io/posts/deep%20learning/2020/04/20/BN.html)

- **Knowledge Distillation**: teacher-student method where a small model is trained to mimic a pre-trained, larger model.
