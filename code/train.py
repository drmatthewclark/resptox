"""
train the basic  model
"""
import torch
import random
from models import SmilesLanguageModelingModel, SmilesLanguageModelingArgs
import os
from model_config import *

model_path = 'outputs/best_model'
train_file = 'chembl_train' 
eval_file = 'chembl_eval'
vocab  = './vocab.txt'
output_dir = 'mlm_model'
epochs = 8
max_size =  1500
batch_size =  8 


args = {
        'config': config, 
        'vocab_path': vocab, 
        'num_train_epochs' : epochs,
        'train_batch_size': batch_size,
        'fp16': True,
        'max_seq_length': max_size,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'output_dir': output_dir,
        'use_multiprocessing' : True,
        'reprocess_input_data' : True,
        'process_count' : 5,
        'optimizer' : 'AdamW',
        'silent' : False,
       }


# create random subsets for training
def random_subset(source_file, random_sample, fraction):

    lines = []
    over = 0

    with open(source_file, 'r') as source:
        for i, smiles in enumerate(source):
            if len(smiles) < max_size and random.random() < fraction :
                lines.append(smiles)
            if len(smiles) > max_size: 
                over += 1

    random.shuffle(lines)
    with open(random_sample, 'w') as out:
        for line in lines:
           out.write(line)

    print('over length', over)
    return len(lines)




if __name__ == "__main__":    

	try:
	    # reads model from path provided to "model_name"
	    model = SmilesLanguageModelingModel(model_type='bert', model_name=model_path, args=args, use_cuda=True)
	    print('read model from file', model_path)
	except:
	    model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args, use_cuda=True)
	    model.save_model()
	    print('created new model')

	model.args.learning_rate = 1e-5/.9

	model.save_model()
	print('start training')
	# train in randomly selected subsets of full set because the full set is too
	# large for the sytem to use
	
	for i in range(100):
	    model.args.learning_rate *= 0.9
	    print('started cycle', i , 'learning rate',  model.args.learning_rate )
	    train = 'chmfp-train'
	    efile = 'chmfp-eval' 
	    fraction = 0.10
	    train_size = random_subset(train_file, train, fraction) 
	    eval_size  = random_subset(eval_file,  efile, fraction)
	    print('train', train_size, 'eval', eval_size )
	    model.train_model(train_file=train, eval_file=efile)
	    result = model.eval_model(efile)
	    model.save_model()
	    print(result)
	    print('finished cycle', i )
	
