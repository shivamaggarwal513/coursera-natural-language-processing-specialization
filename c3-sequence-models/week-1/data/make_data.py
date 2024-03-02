# aclImdb folder must be present in current directory before running
# WARNING: destructive action (deletes some files inside aclImdb folder)
import os
import shutil
import random


def make_data(train_pos, train_neg, test_pos, test_neg):
    data_path = 'aclImdb'
    
    def rm_file_or_dir(file):
        shutil.rmtree(file) if os.path.isdir(file) else os.remove(file)
    
    # [test, train, *]
    for split in os.listdir(data_path):
        data_split_path = os.path.join(data_path, split)
        
        if split not in ['test', 'train', 'README']:
            rm_file_or_dir(data_split_path)
        
        elif os.path.isdir(data_split_path):
            # [neg, pos, *]
            for d_class in os.listdir(data_split_path):
                data_split_class_path = os.path.join(data_split_path, d_class)
                
                if d_class not in ['neg', 'pos']:
                    rm_file_or_dir(data_split_class_path)
                
                else:
                    random.seed(7)
                    sample_names = os.listdir(data_split_class_path)
                    keep_samples = random.sample(sample_names, k=eval(f"{split}_{d_class}"))
                    for sample in sample_names:
                        if sample not in keep_samples:
                            rm_file_or_dir(os.path.join(data_split_class_path, sample))


if __name__ == '__main__':
    print('Input number of samples you want for:')
    train_pos = int(input('\ttrain_pos: ')) # 1000
    train_neg = int(input('\ttrain_neg: ')) # 1000
    test_pos  = int(input('\ttest_pos : ')) # 1000
    test_neg  = int(input('\ttest_neg : ')) # 1000
    
    make_data(train_pos, train_neg, test_pos, test_neg)
