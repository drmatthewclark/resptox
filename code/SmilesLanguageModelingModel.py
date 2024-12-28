import os
import numpy as np
import logging
import torch
import random
from transformers import (
    BertConfig, BertForMaskedLM,
)

wandb_available = False

from SmilesTokenization import SmilesTokenizer
logger = logging.getLogger(__name__)


from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs
)

class SmilesLanguageModelingArgs(LanguageModelingArgs):
    vocab_path: str  =  'vocab.txt'



class SmilesLanguageModelingModel(LanguageModelingModel):

    def _load_model_args(self, input_dir):
        args = SmilesLanguageModelingArgs()
        args.load(input_dir)
        return args


    def select_gpu(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
        if os.environ.get('FORCE_GPU', False):
            return os.environ['FORCE_GPU']
        else:
            return device


    def __init__(
        self,
        model_type,
        model_name,
        generator_name=None,
        discriminator_name=None,
        train_files=None,
        args = None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a LanguageModelingModel.
        Main difference to https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/classification/classification_model.py
        is that it uses a SmilesTokenizer instead of the original Tokenizer.
        Args:
            model_type: The type of model bert (other model types could be implemented)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            generator_name (optional): A pretrained model name or path to a directory containing an ELECTRA generator model.
            discriminator_name (optional): A pretrained model name or path to a directory containing an ELECTRA discriminator model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            train_files (optional): List of files to be used when training the tokenizer.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForMaskedLM, SmilesTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, SmilesLanguageModelingArgs):
            self.args = args


        
        self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.local_rank != -1:
            logger.info(f"local_rank: {self.args.local_rank}")
            torch.distributed.init_process_group(backend="nccl")
            cuda_device = self.args.local_rank

        self.device = self.select_gpu() 

        self.results = {}

        self.args.model_name = model_name
        self.args.model_type = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer_class = tokenizer_class
        new_tokenizer = False

        if self.args.vocab_path:
            self.tokenizer = tokenizer_class(self.args.vocab_path, do_lower_case=False)

        else:
            if not train_files:
                raise ValueError(
                    "model_name and tokenizer_name are not specified."
                    "You must specify train_files to train a Tokenizer."
                )
            else:
                self.train_tokenizer(train_files)
                new_tokenizer = True

        if self.args.config_name:
            self.config = config_class.from_pretrained(self.args.config_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name and self.args.model_name != "electra":
            self.config = config_class.from_pretrained(model_name, cache_dir=self.args.cache_dir, **kwargs)
        else:
            self.config = config_class(**self.args.config, **kwargs)
        if self.args.vocab_size:
            self.config.vocab_size = self.args.vocab_size
        if new_tokenizer:
            self.config.vocab_size = len(self.tokenizer)

        if self.args.block_size <= 0:
            self.args.block_size = min(self.args.max_seq_length, self.tokenizer.model_max_length)
        else:
            self.args.block_size = min(self.args.block_size, self.tokenizer.model_max_length, self.args.max_seq_length)

        if self.args.model_name:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, cache_dir=self.args.cache_dir, **kwargs,
                )
        else:
            logger.info(" Training language model from scratch")
            self.model = model_class(config=self.config)
            model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
            model_to_resize.resize_token_embeddings(len(self.tokenizer))


