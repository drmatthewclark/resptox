import sys
import os
import sklearn
from SmilesClassificationModel import SmilesClassificationModel
from SmilesTokenization import SmilesTokenizer
import torch
import pandas as pd
import random
import numpy as np
import re
import math
from transformers import logging
logging.set_verbosity_error()

fname='training_data.csv'
tune_epochs = 16
output_dir = 'model_dir'
model_path_str = 'outputs/best_model'
report_file = 'crossvalidation.csv'
USE_LOG =  True
learn_rate = 5e-6
batch_size = 16 
saved_model = sys.argv[1]  # pretrained model to fine-tune

def get_model(clean=True):

   if clean:
      os.system('rm -rf model_dir cache_dir mlm_model runs outputs'  )
      os.system('tar -xvf ' + saved_model )
  
   model = SmilesClassificationModel(
       model_type="bert", 
       tokenizer_type=SmilesTokenizer,
       model_name=model_path_str, 
       use_cuda=torch.cuda.is_available(),
       num_labels = 1,
   )

   print('read model', model_path_str )
   with open(model_path_str + '/eval_results.txt', 'r') as f:
         print(f.read())

   model.args.num_train_epochs = tune_epochs
   model.args.regression = True

   # this because multiprocessing is broken 
   model.args.use_multiprocessing =  False
   model.args.use_multiprocessing_for_evaluation = False
   model.args.train_batch_size = batch_size
   model.args.gradient_accumulation_steps=1
   #model.args.gradient_checkpointing=True

   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   #print('parameters', model.config )

   return model

def get_train_eval(model, fname, stats=False):

    frac=0.20
    df = pd.read_csv(fname, sep= '|' )
    df = df.sample(frac = 1.0 )  # shuffle
    if USE_LOG:
        df['labels'] = np.log10( df['labels']  + 1e-5 )

    # some smiles may be greater than the model max size.  drop them
    maxlen = model.args.config['max_position_embeddings']
    maxseq = model.args.max_seq_length
    maxlen = min(maxlen, maxseq) 

    df['xx'] = df.text.str.len()
    start_size = len(df)
    df = df[ df.xx < maxlen ]       # drop over-length
    end_size = len(df)
    print('dropped %3d over length SMILES' % (start_size-end_size,))
    df = df.drop(['xx'], axis = 1)  # drop the length column

    if stats:
       maxval = max(df['labels'])
       minval = min(df['labels'])
       rangev = maxval - minval
       aveval = np.mean(df['labels'])
       print('data min %5.2f max %5.2f range %5.2f mean %5.2f size %d' % (minval, maxval, rangev,aveval, len(df) ) )

    # separate into test set/train set
    length = len(df)
    test = int(length*frac)
    df = df.sample(frac = 1.0 ) # re-randomize
    train_df = df.iloc[test:].copy()
    eval_df  = df.iloc[:test].copy()
    print('train set %d eval set %d fraction %f' % (len(train_df), len(eval_df), len(eval_df)/(len(train_df) + len(eval_df))) )

    #train_df = pd.concat( [train_df, train_df, train_df] ) # multiply x2for training
    #train_df = train_df.sample(frac=1.0)

    return train_df, eval_df

def write(model, train_df, eval_df, cycle):
   
   train_preds = model.predict(list(train_df.text))
   eval_preds  = model.predict(list(eval_df.text))
  
   train_r2 = np.corrcoef(train_df['labels'], train_preds[0] )[0][1]
   eval_r2  = np.corrcoef(eval_df['labels'], eval_preds[0] )[0][1]

   print('train r2 %5.3f  eval r2 %5.3f\n\n\n' % (train_r2, eval_r2,)  )
  
   train_df['pred'] = train_preds[0]
   eval_df['pred'] = eval_preds[0]
   train_df.to_csv('train_results%03d.csv' % (cycle+1,) )
   eval_df.to_csv('eval_results%03d.csv' % (cycle+1,)  )
  
 

if __name__ == '__main__':  

   loss = []
   mvalue = [] 
   metric = sklearn.metrics.r2_score
   cycles = 32 
   xmodel = None

   for i in range(cycles):
       print('cycle', i+1) 

       xmodel = get_model(clean=True)
       xmodel.args.learning_rate = learn_rate
       train_df,eval_df  = get_train_eval(xmodel, fname, stats = True)

       print('learning_rate %9.8f' % ( xmodel.args.learning_rate) )
       xmodel.train_model( train_df,  eval_df = eval_df, acc= metric )
       xmodel = get_model(clean=False)  # get best model of series

       tresult, _, _  = xmodel.eval_model(train_df, acc =  metric, silent=True, verbose=False )
       print('train result   %6d  acc %6.3f eval_loss %6.3f' % ( len(train_df), tresult['acc'], tresult['eval_loss'] ) )

       result, _, _ = xmodel.eval_model(eval_df, acc =  metric, silent=True, verbose=False )
       print('eval  result   %6d  acc %6.3f eval_loss %6.3f' % ( len(eval_df), result['acc'], result['eval_loss'] ) )

       loss.append( math.sqrt( result['eval_loss' ] ) ) # sqrt for rms
       mvalue.append( result['acc'] )  # r-squared 
      
       with open(report_file, 'a') as file:
           val = '%d r2 %f rms error %f  ave rms %f stdev rms %f npoints %d\n' % (i, result['acc'], math.sqrt(result['eval_loss']), np.mean(loss), np.std(loss), len(loss)  ) 
           print(val)
           file.write( val )

       write(xmodel, train_df, eval_df, i ) 
       print('\n')
   
   
   
   
