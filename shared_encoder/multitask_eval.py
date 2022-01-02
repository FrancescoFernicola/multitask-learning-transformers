from multitask_data_collator import DataLoaderWithTaskname
import nlp
import numpy as np
import torch
import transformers
from datasets import load_metric
import pandas as pd

def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    metric = load_metric("rmse")
    
    for idx, task_name in enumerate(["BERTScore", "CushLEPOR", "COMET", "TransQuest"]):
        val_len = len(features_dict[task_name]["validation"])
        eval = 0
        for index in range(0, val_len, batch_size):

            batch = features_dict[task_name]["validation"][index : min(index + batch_size, val_len)]["doc"]
            
            labels = features_dict[task_name]["validation"][index : min(index + batch_size, val_len)]["target"]
            inputs = tokenizer(batch, max_length=512)
            
            inputs["input_ids"] = torch.LongTensor(inputs["input_ids"]).cuda()
            inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"]).cuda()
            
            logits = multitask_model(task_name, **inputs).logits
            
            predictions = torch.argmax(torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),dim=1)           
                     
            metric.add_batch(predictions=predictions, references=np.array(labels))

            print(f"\nRMSE value for current batch: {metric.compute()['rmse']}")
            
            eval += metric.compute()["rmse"]
            
            print("\nCurrent total RMSE value: {eval}")
        
        eval = eval/val_len
        preds_dict[task_name] = eval
        print(f"\nTask name: {task_name}\tFinal RMSE: {eval}\n\n")
    
    preds = pd.DataFrame.from_dict(preds_dict)
    print(preds)
