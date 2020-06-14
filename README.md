# SPHERE NORM：Make GCN Deep Again

1. Node classification : 

   Dataset: Cora,  Pubmed and Citeseer
   
   Model：GCN/GAT/GIN

​	  use command line  python train.py --dataset Cora --model gcn/gat/gin  to run the code

2. Graph classification: 

   Dataset: D&D, PROTEINS and ENZYMES

   Model: GCN/GAT/GIN

   use command line python train.py --dataset 

   ​                                                               --model

   ​                                                               --norm_type  sp_norm(sphere norm)/b_norm(batch norm)/res

   ​                                                               --hidden 128(for GCN/GIN) / 16 (for GAT)

   ​                                                               --learning_rate 7e-4

   ​                                                               --layers (default 4 layers)

   to run the model

3. SPHERE GCN:

   Dataset: Cora, Pubmed and Citeseer

   include SphereGCN/BnResGCN model and DenseGCN model

   use command line python train.py --dataset

   ​                                                               --model

   ​                                                               --norm_type sp_norm/b_norm/res

   to run the model

   

   

   