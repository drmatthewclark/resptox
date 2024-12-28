from simpletransformers.config.model_args import ClassificationArgs
from simpletransformers.classification import ClassificationModel
from transformers import BertConfig, BertForSequenceClassification
from simpletransformers.losses.loss_utils import init_loss        
from SmilesTokenization import SmilesTokenizer

import torch
import random
import numpy as np
import os

from simpletransformers.losses.loss_utils import init_loss        
from SmilesTokenization import SmilesTokenizer


class SmilesClassificationModel(ClassificationModel):
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=SmilesTokenizer,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        freeze_encoder=False,
        freeze_all_but_one=False,
        **kwargs,
    ):


        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, SmilesTokenizer),
        }

        if model_type not in MODEL_CLASSES.keys():
            raise NotImplementedError(f"Currently the following model types are implemented: {MODEL_CLASSES.keys()}")

        self.args = self._load_model_args(model_name)
        self.use_cuda =  use_cuda
        self.tokenizer = SmilesTokenizer('vocab.txt')
        self.weight = weight
        self.cuda_device = cuda_device
 
        self.device = 'cuda'

        self.loss_fct = init_loss(
            weight=self.weight, device=self.device, args=self.args
        )

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        
        self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.labels_list:
            if num_labels:
                assert num_labels == len(self.args.labels_list)
            if self.args.labels_map:
                try:
                    assert list(self.args.labels_map.keys()) == self.args.labels_list
                except AssertionError:
                    assert [
                        int(key) for key in list(self.args.labels_map.keys())
                    ] == self.args.labels_list
                    self.args.labels_map = {
                        int(key): value for key, value in self.args.labels_map.items()
                    }
            else:
                self.args.labels_map = {
                    label: i for i, label in enumerate(self.args.labels_list)
                }
        else:
            len_labels_list = 2 if not num_labels else num_labels
            self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        self.weight = weight


        if self.use_cuda:
            if torch.cuda.is_available():
                if self.cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            print('using CPU to train')
            self.device = "cpu"

        self.results = {}

        if tokenizer_name is None:
            tokenizer_name = model_name

        self.model = model_class.from_pretrained(
                        model_name, config=self.config, **kwargs
                    )

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type
        self.args.wandb_project = None

