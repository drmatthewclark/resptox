# model definition; head, layers, etc

config = {
  "architectures": [ "BertForMaskedLM" ],
  "model_type": "bert",
  "hidden_act": "gelu",
  "hidden_size": 256,
  "intermediate_size":  512,
  "max_position_embeddings": 512 ,
  "num_attention_heads": 8,
  "num_hidden_layers": 16,
  "attention_probs_dropout_prob": 0.1,
  "hidden_dropout_prob": 0.1,
}

