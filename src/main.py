import pickle
from haystack.nodes import TfidfRetriever
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from config import Finetuned_LM_Path, KG_Graph_File, KG_Embeddings_File

# Helper functions that use graph embeddings
def get_similar_nodes(node, embeddings, topn):
    similar_nodes = embeddings.wv.most_similar(node, topn = topn)
    return similar_nodes

def get_content_from_embeddings(graph, node, embeddings):
    similar_nodes = embeddings.wv.most_similar(node, topn = 10)
    content = str(node)
    for n in similar_nodes:
        add_context = str(n[0])
        # if graph.has_edge(node, n[0]):
        #     #print('Edge between ' + str(n[0]) + ' and ' + str(node))
        #     edge_attributes = graph.get_edge_data(node, n[0])
        #     relation = edge_attributes[0]['Relation']
        #     if relation == 'directing':
        #         add_context = add_context + ' is directed by ' +  str(n[0])
        #     elif relation == 'acting':
        #         add_context = add_context + ' has acted in ' +  str(n[0])
        #     elif relation == 'genre':
        #         add_context = add_context + ' is a ' +  str(n[0])
        #     elif relation == 'released':
        #         add_context = add_context + ' is a ' +  str(n[0])
        content = content + ' ' + add_context
    return content

# Helper functions for question answering to postprocess and return top k
def get_answer_from_question(pipe, query, top_k):
    preds = pipe.run(query = query, params={"Retriever": {"top_k": top_k}, "Reader" : {"top_k" : top_k}})
    ans = postprocess_answers(preds, top_k)
    return ans    

def postprocess_answers(answers, top_k):
    final_ans = []
    graph, _ = get_graph_and_embeddings(KG_Graph_File, KG_Embeddings_File)
    for ans in answers['answers']:
        ans_seq = ans.answer
        for node in graph.nodes():
            if node in ans_seq and node not in final_ans:
                final_ans.append(node)
    return final_ans[:top_k]

# import graph and embeddings from pkl file
def get_graph_and_embeddings(KG_Graph_File, KG_Embeddings_File):
    with open(KG_Graph_File, 'rb') as f:
        graph = pickle.load(f)
    with open(KG_Embeddings_File, 'rb') as f:
        embeddings = pickle.load(f)
    return graph, embeddings

# Return the LLM pipeline whihc merges KG and LM to return context for QA tasks
def get_qa_pipeline(KG_Graph_File, KG_Embeddings_File, Finetuned_LM_Path):
    # Design a retriever for LLM that will retireve docs from query
    # Step 1: Creating custom docs from graph nodes and embeddings
    custom_documents = []
    graph, embeddings = get_graph_and_embeddings(KG_Graph_File, KG_Embeddings_File)
    for node in graph.nodes():
        doc = {}
        doc['content'] = get_content_from_embeddings(graph, node, embeddings)
        custom_documents.append(doc)
    # Step 2: Create an InMemoryDocumentStore and add your custom documents
    document_store = InMemoryDocumentStore()
    document_store.write_documents(custom_documents, )
    # Step 3: Create a custom retriever
    retriever = TfidfRetriever(document_store = document_store)
    # Use finetuned LM instead of pretrained one
    reader = FARMReader(model_name_or_path = Finetuned_LM_Path, use_gpu = True)
    # Merge KG Retriever and LM to create pipeline
    pipe = ExtractiveQAPipeline(reader, retriever)
    return pipe


# Driver Code
if __name__ == "__main__":
    pipeline =  get_qa_pipeline(KG_Graph_File, KG_Embeddings_File, Finetuned_LM_Path)
    sample_question = "Which film stars Leonardo DiCaprio and was released in 2015?"
    top_k = 10
    print(get_answer_from_question(pipeline, sample_question, top_k))




