# NaLP: Link Prediction on N-ary Relational Data

This project provides the tensorflow implementation of the link prediction model NaLP on n-ary relational data, published in WWW'19.

## Usage
### Prerequisites
- Python 3.6
- Tensorflow 1.4.0

### Prepare data
Transform the representation form of facts for [JF17K](https://github.com/lijp12/SIR). Transform each value sequence in JF17K to a set of role-value pairs:

    python JF17K2rv_json.py

Build data before training and test for JF17K and [WikiPeople](https://github.com/gsp2014/WikiPeople):

    python builddata.py --sub_dir JF17K_version1 --dataset_name JF17K_version1
    python builddata.py --sub_dir WikiPeople --dataset_name WikiPeople

Build data for filtering the right facts in negative sampling or computing the filtered metrics when evaluation:

    python builddata.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --if_permutate True --bin_postfix _permutate
    python builddata.py --sub_dir WikiPeople --dataset_name WikiPeople --if_permutate True --bin_postfix _permutate

### Training
To train NaLP:

    python train_only.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --wholeset_name JF17K_version1_permutate --model_name JF17K_version1_opt --embedding_dim 100 --n_filters 200 --n_gFCN 1000 --batch_size 128 --learning_rate 0.00005 --n_epochs 5000 --saveStep 100
    python train_only.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --n_filters 200 --n_gFCN 1200 --batch_size 128 --learning_rate 0.00005 --n_epochs 5000 --saveStep 100
            
### Evaluation
Files `see_eval.py` and `see_eval_bi-n.py` provide four evaluation metrics, including the Mean Reciprocal Rank (MRR), Hits@1, Hits@3 and Hits@10 in filtered setting. In these two files, parameter **--valid_or_test** indicates whether to evaluate NaLP in the validation set (set to 1) or test set (set to 2).

To evaluate NaLP in the validation set (JF17K lacks a validation set):

    python see_eval.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --n_filters 200 --n_gFCN 1200 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 1 --gpu_ids 0,1,2,3

To evaluate NaLP in the test set:

    python see_eval.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --wholeset_name JF17K_version1_permutate --model_name JF17K_version1_opt --embedding_dim 100 --n_filters 200 --n_gFCN 1000 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 2 --gpu_ids 0,1,2,3
    python see_eval.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --n_filters 200 --n_gFCN 1200 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 2 --gpu_ids 0,1,2,3

File `see_eval_bi-n.py` provides more detailed results on binary and n-ary relational facts. It is used in the same way as `see_eval.py`.

Note that, it takes a lot of time to evaluate NaLP, since we need to compute a score via NaLP for each candidate (each value/role in the value/role set). To speed up the evaluation process, `see_eval.py` and `see_eval_bi-n.py` are implementated in a multi-process manner.

## Citation
If you found this codebase or our work useful please cite:

    @inproceedings{NaLP,
      title={Link prediction on n-ary relational data},
      author={Guan, Saiping and Jin, Xiaolong and Wang, Yuanzhuo and Cheng, Xueqi},
      booktitle={Proceedings of the 28th International Conference on World Wide Web (WWW'19)},
      year={2019},
      pages={583--593}
    }

## Related work
[A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://github.com/daiquocnguyen/ConvKB)
