from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import forward

def TestForward(jit_path):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased', torchscript=True)
  model.cpu()
  model.eval()

  inputs = tokenizer("Hello, my dog is cute", max_length=128, pad_to_max_length=True, return_tensors="pt")
  dummy_inputs = (
    inputs["input_ids"],
    inputs["attention_mask"],
    inputs["token_type_ids"],
  )
  
  print('dummy_inputs :', dummy_inputs)

  traced_model = torch.jit.trace(model, dummy_inputs)
  traced_model.save(jit_path)
  print('Jit model is saved.')

  builder = forward.TorchBuilder()
  builder.set_mode('float32')
  engine = builder.build(jit_path, dummy_inputs)
  
  engine_path = jit_path + '.engine'
  engine.save(engine_path)

  engine = forward.TorchEngine()
  engine.load(engine_path)

  ground_truth = traced_model(*dummy_inputs)
  print('ground_truth', ground_truth)
  outputs = engine.forward(dummy_inputs)
  print('outputs : ', outputs)

if __name__ == "__main__":
  jit_path = 'bert_cpu.pt'
  print("jit_path : ", jit_path)
  TestForward(jit_path)
