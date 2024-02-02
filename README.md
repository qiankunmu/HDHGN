# Heterogeneous Directed Hypergraph Neural Network (HDHGN)

## Introduction
This is the implementation code of Heterogeneous Directed Hypergraph Neural Network (HDHGN) model in the paper [Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification](https://doi.org/10.18293/SEKE2023-136). Our paper is accepted by the 35th International Conference on Software Engineering and Knowledge Engineering (SEKE 2023) as a regular paper. You can use this code to reappear the results of our model in the paper.

## Requirments
Our experiment is done in Ubuntu 18.04.6 LTS and in python 3.8. We implement our model by [pytorch 1.10](https://pytorch.org/docs/1.10/) and [torch geometric 2.1.0](https://pytorch-geometric.readthedocs.io/en/2.1.0/index.html). We train our model on a RTX 3090. The required environments are listed in requirements.txt.

## Datasets
Our datasets Python800 and Java250 are from [Project CodeNet](https://github.com/IBM/Project_CodeNet). You can download the datasets from <https://developer.ibm.com/data/project-codenet/>. You need to download [Project_CodeNet_Python800.tar.gz](https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz) and [Project_CodeNet_Java250.tar.gz](https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Java250.tar.gz). Then you need to put them into the data directory of our project.

## Usages
You can directly run run.sh to reappear the results of our model.
```
sh run.sh
```
run.sh will do the followings:
1. Unzip python800 and java250 datasets.
2. Run ProcessData.py to randomly split data into train, valid, test set by 6:2:2. It doesn't directly split the data, it will record the corresponding paths instead.
3. Run vocab.py to generate vocab files.
4. Run trainHDHGN.py to train model on python800 and test the accuracy result after training.
5. Run trainHDHGN_java.py to train model on java250 and test the accuracy result after training.

If you don't want to use run.sh to complete all the operations at once, you can also perform them step by step by yourself as follows.
1. Enter our project directory and enter the data folder. Then unzip python800 and java250 datasets.
```
cd data
tar -zxvf Project_CodeNet_Java250.tar.gz
tar -zxvf Project_CodeNet_Python800.tar.gz
```
2. Then return to the previous directory. Run ProcessData.py, it will split python800 and java250 separately into train, valid, test set in order.
```
cd ..
python ProcessData.py
```
3. Then run vocab.py, it will generate vocab.json and vocab_java.json for python and java datasets.
```
python vocab.py
```
4. Then you can enter trains directory. You can run trainHDHGN.py to train model on python800.
```
cd trains
python trainHDHGN.py
```
5. You can run trainHDHGN_java.py to train model on java250.
```
python trainHDHGN_java.py
```
6. If you want to re-train model multiple times, you can run trainXXX.py directly. When you first train model on python or java, it will take time to process data into tensor format and save them, and it will take no time to process data when you train model next time. If you want to re-split datasets, you need to run step2 and step3 again before training. You also need to delete the tensor format data as follows.
```
cd data
rm -rf train valid test train_java valid_java test_java
```
7. After training, program will test the model on test set and show the accuracy results. The model will be saved into work_dir directory. The result will also be saved into work_dir/results.xlsx. The figures of changes in training losses and valid set results will also be saved into work_dir/HDHGN/XXX-loss.png and work_dir/HDHGN/XXX-accuracy.png.

If you want to modify hyper-parameters such as embed_size, dim_size, learning_rate, batch_size, num_epochs, you can modify them in trainXXX.py by yourself.

We didn't test it on other machines, so you may meet environmental problems. You can consult relevant websites or contact us. 

## Citation
If you use our code, please cite us.
```
@inproceedings{Yang2023HeterogeneousDH,
  title={Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree {(AST)} for Code Classification},
  author={Guang Yang and Tiancheng Jin and Liang Dou},
  booktitle={The 35th International Conference on Software Engineering and Knowledge Engineering, {SEKE} 2023},
  year={2023},
  doi={10.18293/SEKE2023-136}
}
```

