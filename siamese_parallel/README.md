# Parallelized Prediction Process
* Node represents each node in the network. It can support different models on different
machine according to request from previous node. 
* Initial send out original data to other nodes in the network.

## How To Run
run on the localhost
```buildoutcfg
python node.py
```
run on different machines
```buildoutcfg
python node.py -a 8.8.8.8 -p 12345
```

specify dimension for fully connected layer
```buildoutcfg
python node.py -dim 1024
```

use flag for debug
```buildoutcfg
python node.py -d
```