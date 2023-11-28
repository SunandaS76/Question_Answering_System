# Question_Answering_System




### Configuration
Set this up in the src/config.py

###### KG Config : To create them you need to run KG notebook in notebook folder
KG_Graph_File = '../models/graph.pkl',
KG_Embeddings_File = '../models/embeddings.pkl'

###### LM Config : Use either pretrained or finetuned. To finetune it use the LM Finetuning notebook from notebook folder
Finetuned_LM_Path = 'deepset/roberta-base-squad2',
Finetuned_LM_Path = 'DKud7/finetuned-roberta-squad2'

###### All models and data are available in github repo releases section

___

### How to Run ?

Use the following to answer a single question
```
python src/main.py
```

Use the following to answer multiple questions from test file and generate top k answers for each question. 
It will also print the overall accuracy
```
python src/test.py
```

___
