from main import get_qa_pipeline, get_answer_from_question
from config import KG_Graph_File, KG_Embeddings_File, Finetuned_LM_Path



## Read 20 Questions from test file
test_file = '../data/test.txt'
answers_file = '../data/answers.txt'
questions = []
true_answers = []
topk_predicted_answers = []

with open(test_file, 'r') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        if i % 2 == 0 and i < 39:
            questions.append(line.strip())
        elif i > 41 and i < 62:
            true_answers.append(line.strip()[8:-1].lower())
        else:
            continue

# Begin inference
pipe = get_qa_pipeline(KG_Graph_File, KG_Embeddings_File, Finetuned_LM_Path)
topk = 7

# Write answers in file
g = open(answers_file, 'w+')
N = len(questions)
for i in range(N):
    g.write('QUESTION: ' + questions[i] + '\n')
    g.write('TRUE ANSWER: ' + true_answers[i] + '\n')
    preds = get_answer_from_question(pipe, questions[i], topk)
    g.write('PREDICTIONS: ' + ','.join(preds) + '\n\n')

g.close()