# multiRelational GraphStar
Adaption of the [GraphStar Net](https://arxiv.org/pdf/1906.12330v1.pdf) architecture to predict links in multiple-relation datasets


## Introduction

The main issue with LP remains the quality of the inferred missing triples. A serious weakness in early deep learning approaches stemmes from the finding that convolutional neural networks struggle with prediction tasks in large complex graphs. This is a result of the difficulties of defining convolutional and pooling layers for non-Euclidean data. This, in turn, inspired the rise of Graph Neural Networks(GNN), which is currently a key research area in the AI community. A promising example is the GraphStar Net architecture. Information propagation is the key idea that contributes to the recent success of GNNs on various tasks. However, there still lacks an efficient framework that is able to achieve global message passing across the whole graph. GraphStar presents a novel and unified graph neural network architecture that achieves non-local graphical representation through topological modification of the original graph. In particular, they propose to use a virtual “star” node to propagate global information to all real nodes, which significantly increase the representation power without increasing the model depth or bearing heavy computational complexity.  

In GraphStar, the authors have implemented a link prediction model for the special case of single-relation datasets. Though applicable to many problems, several tasks such as particular content based fake news detection systems require models to predict links in multiple-relation datasets.

--------------------------------------------------------------------------------
## Datasets

The datasets used in this implementation are FB15k and FB15k-237. They are selected as they have been used extensively in prior work on LP. To this date, FB15k is one of the most commonly used benchmark for LP. This enables easy and comprehensive comparisons between our model and previous work.


--------------------------------------------------------------------------------
## Dependencies

The script has been tested under Python 3.6.X and on Nvidia GPU Tesla V100 (with CUDA 9.0 and cuDNN 7 installed). For installing required python packages, please refer to the "requirements.txt".
 
The code will not work using newer versions of Python, such as Python 3.8.x  
We are currently running [Python 3.7.5](https://www.python.org/downloads/release/python-375/)

--------------------------------------------------------------------------------
## Options
 Normally we have already provided most of the optimal value which we got during traning. So you don't have to change much. Any away, we still explain the main options here.

 ```
   --device                                     INT/STR GPU ID / cpu                               Default is `0`, use 'cpu' if no GPU is available
   --num_star                                   INT     Number of Star.                                Default is 1.
   --epochs                                     INT     Number of training epochs.                     Default is 2000.
   --lr                                         FLOAT   Adam learning rate.                            Default is 2e-4.
   --dropout                                    FLOAT   Dropout rate value.                            Default is 0.0.
   --coef_dropout                               FLOAT   Dropout for attention coefficients.            Default is 0.0.
   --num_layers                                 INT     Number of layers.                              Default is 6.
   --hidden                                     INT     Number of hidden units.                        Default is 1024.
   --heads                                      INT     Multi-head for Attention.                      Default is 4.
   --l2                                         FLOAT   Regularization.                                Default is 0.
   --residual                                   STR     Skip connections across attentional layer      Default is True.
   --residual_star                              STR     Skip connections across attentional star       Default is True.
   --layer_norm                                 STR     Layer normalization                            Default is True.
   --layer_norm_star                            STR     Layer normalization for star                   Default is True.
   --activation                                 STR     Activation methods                             Default is `elu`.
   --star_init_method                           STR     Star initialization method                     Default is `attn`.
   --additional_self_loop_relation_type         STR     As its name implied                            Default is True.
   --additional_node_to_star_relation_type      STR     As its name implied                            Default is True.
   --relation_score_function                    STR     Score function for relation                    Default is `DistMult`.
   --patience                                   INT     Number of patience                             Default is 100.
   --dataset                                    STR     Name of dataset                                Default is "FB15k_237".
 ```

--------------------------------------------------------------------------------

## Setup
 
 1) Create a virualenv using Python 3.7.x
 
 ``` 
 python3.7 -m virtualenv env37 
 source env37/bin/activate
 ```
 
 2) Install dependencies
 
 ```
 python setup.py cu110
 ```

Here you can change cu110 to the prefered version. Possible versions are `cpu`, `cu92`, `cu101`, `cu102` and `cu110`.


## Testing (Run the code)

 1) For link prediction tasks (FB15k, FB15k-237), choose related script then try
 ```sh
python run_mr_lp.py --dropout=0 --hidden=256 --l2=5e-4 --num_layers=3 --cross_layer=False --patience=200 --residual=True --residual_star=True --dataset=FB15k_237 --device=cpu --epochs=50
 ```


--------------------------------------------------------------------------------
## Tensorboard Visualization

To visualize model metrics such as loss and accuracy, we log these values using tensorboardX's SummaryWriter. 
To visualize the metrics of your model run:

```tensorboard --logdir="./tensorboard"```

## Performance 
Combined average of AUC and AP scores. CPC denotes the average performance onthe three datasets; Cora, Pubmed, and Citseer

| Models  | CPC  | FB15k  |  FB15k-237
| :------------ | :------------ | :------------ | :------------|
| GraphStar  | 0.969  |   |   |
| **mrGraphStar**  |   | 0.928  | 0.983  |

## Performance of original architecture

Below are the results presented by the [graph_star_team](https://github.com/graph-star-team). 

### Node Classification (Transductive)
| Models  | Cora (Acc)  | Citeseer (Acc)  | Pubmed (Acc)
| :------------ | :------------ | :------------ | :------------|
| GCN  | 0.815  | 0.703  | 0.790  |
| GAT  | 0.830  | 0.725  | 0.790  |
| SGC  | 0.806  | 0.714  | 0.770  |
| MTGAE  | 0.790  | 0.718  | **0.804**  |
| LGCN  | **0.833**  | **0.730**  | 0.795  |
| **GraphStar**  | 0.821(0.012)  | 0.71(0.021)  | 0.772(0.011)  |

### Node Classification (Inductive)

| Models  | PPI (F1-Micro)  |
| :------------ | :------------ |
| GraphSage  | 0.612  |
| GAT  | 0.973  |
| LGCN  |  0.772  |
| JK-Dense LSTM  | 0.500  |
| GaAN  | 0.987  |
| **GraphStar**  | **0.994(0.001)**  |

| Models  | IMDB (Acc)  |
| :------------ | :------------ |
| oh-LSTM  | 0.941  |
| L Mixed  | 0.955  |
| BERT large finetune UDA  | 0.958  |
| **GraphStar**  | **0.960(0.001)**  |


### Graph Classification
| Models  | Enzymes  | D&D  | Proteins  | Mutag  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
| Seal-Sage  | -  | **0.809**  | 0.772  | -  |
| Diff Pool  | 0.625  | 0.806  | 0.763  | -  |
| CAPS GNN  | 0.574  | 0.754  | 0.763  | 0.867  |
| **GraphStar**  | **0.671(0.0027)**  | 0.796(0.005)  | **0.779(0.008)**  | **0.912(0.021)**  |

### Graph Text Classification
| Models  | R8  | R52  | 20NG  | MR  | Ohsumed  |
| :------------ | :------------ | :------------ | :------------ | :------------ | :------------ |
| Bi-LSTM  | 0.963  | 0.905  | 0.732  | **0.777**  | 0.493  |
| fastText  | 0.961  | 0.928  | 0.794  | 0.751  | 0.557  |
| TextGCN  | 0.9707  | 0.9356  | 0.8634  | 0.7674  | 0.6836  |
| SGCN  | 0.9720  | 0.9400  | **0.8850**  | 0.7590  | **0.6850**  |
| **GraphStar**  | **0.974(0.002)**  | **0.950(0.003)**  | 0.869(0.003)  | 0.766(0.004)  | 0.642(0.006)  |

### Link Prediction
| Models  | Cora  | Citeseer  | Pubmed  |
| :------------ | :------------ | :------------ | :------------ |
| MTGAE  | 0.946  | 0.949  | 0.944  |
| VGAE  | 0.920  | 0.914  | 0.965  |
| **GraphStar**  | **0.959(0.003)**  | **0.977(0.003)**  | **0.970(0.001)**  | 

--------------------------------------------------------------------------------
<!--
## Reference
If you find this work useful for your research, please consider citing our work in following way after published.



```
@article{
  graphStar2019,
  title="{Graph Star Net for Generalized Multi-Task Learning}",
  author={**},
  journal={NeurIPS},
  year={2019}
}
```
-->
## Acknowledgement
This code is based on the graph_star_team's work, which is based on Pytorch_geometry's work, so we would like to thank all the previous contributors here.
