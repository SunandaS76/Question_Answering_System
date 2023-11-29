# Question_Answering_System
CS 635  - IR
22m2116 - Dhruv Kudale
22m2107 - Sunanda Somwase

### Configuration
Set this up in the src/config.py

###### KG Config: To create them you need to run the KG notebook in the notebook folder
KG_Graph_File = '../models/graph.pkl',
KG_Embeddings_File = '../models/embeddings.pkl'

###### LM Config: Use either pretrained or finetuned. To finetune it use the LM Finetuning notebook from the notebook folder
Finetuned_LM_Path = 'deepset/roberta-base-squad2',
Finetuned_LM_Path = 'DKud7/finetuned-roberta-squad2'

###### All models and data are available in the GitHub repo releases section

___

### How to Run?

Use the following to answer a single question
```
python src/main.py
```

You can use the following to answer multiple questions from the test file and then generate top k answers for each question. 
It will also print the overall accuracy
```
python src/test.py
```
### Results 


| Left |  Center  | Right |
|:-----|:--------:|------:|
| L0   | **bold** | $1600 |
| L1   |  `code`  |   $12 |
| L2   | _italic_ |    $1 |




___
