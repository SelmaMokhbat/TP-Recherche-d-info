import math
from pydoc import doc
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer
from collections import Counter
import re
import math
from flask import Flask, render_template,request
from flask import jsonify


app = Flask(__name__)
#traiter le mot taper avec Regex et Porter

def extract_and_stem(query, method, stemmer):
    tokenizer = None

    if not query or not isinstance(query, str):
        raise ValueError("La requête ne peut pas être vide et doit être une chaîne de caractères")

    if method == 'Token':
        tokenizer = RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
        
    elif method == 'Split':
        tokenizer = RegexpTokenizer(r'\s+')
    else:
        raise ValueError("Méthode d'extraction non prise en charge")

    stemmer_instance = None
    if stemmer == 'Porter':
        stemmer_instance = PorterStemmer()
    elif stemmer == 'Lancaster':
        stemmer_instance = LancasterStemmer()
    else:
        raise ValueError("Stemmer non pris en charge")

    if method == 'Token':
        terms = [stemmer_instance.stem(token.lower()) for token in tokenizer.tokenize(query)]
    elif method == 'Split':
        # Si la méthode est 'Split', divisez simplement la requête en mots
        terms = [stemmer_instance.stem(word.lower()) for word in query.split()]

    return terms


def view_content(method, stemmer, index, query):
    content = []

    if index == "Descripteur":
        result_folder = 'result'
        filename = f'{index}{method}{stemmer}.txt'
        descriptor_file = os.path.join(result_folder, filename)
        with open(descriptor_file, 'r', encoding='utf-8') as descriptor_file:
           for line in descriptor_file.readlines():
               
                if any(any(term.strip() == line_term for line_term in line.strip().split()) for term in query):
                 content.append(line)

    elif index == "Inverse":
        result_folder = 'result'
        filename = f'{index}{method}{stemmer}.txt'
        inverse_file = os.path.join(result_folder, filename)
        with open(inverse_file, 'r', encoding='utf-8') as inv_file:
            for line in inv_file.readlines():
                # Vérifier si la ligne contient exactement la requête
                if any(any(term.strip() == line_term for line_term in line.strip().split()) for term in query):
                 content.append(line)

  

    return content


def scalar_product(query_vector, doc_vector):
    return np.dot(query_vector, doc_vector)

def cosine_measure(query_vector, doc_vector):
    # return scalar_product(query_vector, doc_vector)/math.sqrt(np.sum(np.square(query_vector)))*math.sqrt(np.sum(np.square(doc_vector)))
    norm_query = np.linalg.norm(query_vector)
    norm_doc = np.linalg.norm(doc_vector)
    
    if norm_query == 0 or norm_doc == 0:
        return 0  # Éviter la division par zéro
    
    return np.dot(query_vector, doc_vector) / (norm_query * norm_doc)

def jaccard_measure(query_vector, doc_vector):
    numerator = np.dot(query_vector, doc_vector)
    denominator = np.sum(np.square(query_vector)) + np.sum(np.square(doc_vector)) - numerator
    
    if denominator == 0:
        return 0  # Éviter la division par zéro
    
    return numerator / denominator


def apply_model(model_type, query_terms, method, stemmer, index):
    result_folder = 'result'
    filename = f'{index}{method}{stemmer}.txt'
    file_path = os.path.join(result_folder, filename)

    with open(file_path, 'r') as file:
        data = [line.split() for line in file]

    doc_weights = {}
    doc_vercors = {}

    for line in data:
        if index=='Inverse':
         term, doc_id, frequency, weight = line[0], line[1], line[2], line[3]
        elif index=='Descripteur':
            term, doc_id, frequency, weight = line[1], line[0], line[2], line[3]
        if doc_id not in doc_weights:
            doc_weights[doc_id] = {}
            doc_vercors[doc_id] = {}

        doc_weights[doc_id][term] = float(weight)
        doc_vercors[doc_id][term] = 1 if term in query_terms else 0

    query_vector = np.array([1 if term in query_terms else 0 for term in doc_weights[list(doc_weights.keys())[0]]])

    results = []
    for doc_id in doc_weights.keys():
        doc_vector = np.array([weight for term, weight in doc_weights[doc_id].items()])
        query_vector = np.array([i for term, i in doc_vercors[doc_id].items()])
        
        if model_type == 'Produit Scalaire':
            relevance = scalar_product(query_vector, doc_vector)
        elif model_type == 'Similarité Cosinus':
            relevance = cosine_measure(query_vector, doc_vector)
        elif model_type == 'Indice de Jaccard':
            relevance = jaccard_measure(query_vector, doc_vector)
        else:
            raise ValueError("Modèle non pris en charge")

        results.append((doc_id, relevance))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results


def calculate_relevance(model_type, query_vector, doc_vector):
    if model_type == 'Produit Scalaire':
        return scalar_product(query_vector, doc_vector)
    elif model_type == 'Similarité Cosinus':
        return cosine_measure(query_vector, doc_vector)
    elif model_type=='Indice de Jaccard':
        return jaccard_measure(query_vector,doc_vector)
    else:
        raise ValueError("Modèle non pris en charge")
    

def bm25_score(fi, doc_len, avg_doc_len, K, B, N, ni):
    fi = int(fi)  # Convert fi to integer if it's a string
    doc_len = int(doc_len)  # Convert doc_len to integer if it's a string
    avg_doc_len = float(avg_doc_len)  # Convert avg_doc_len to float if it's a string
    K = float(K)  # Convert K to float if it's a string
    B = float(B)  # Convert B to float if it's a string
    N = int(N)  # Convert N to integer if it's a string
    ni = int(ni)  # Convert ni to integer if it's a string

    return (fi / (K * ((1 - B) + B * (doc_len / avg_doc_len)) + fi)) * math.log10((N - ni + 0.5) / (ni + 0.5))




def modele_BM25(request:list, N, method, stemmer, index):
    freqii = {}
    freqi = [[0] * N for _ in range(len(request))]
    dl = [0] * N
    avdl = 0.0
    ni = [0] * len(request)
    result_folder = 'result'
    filename = f'{index}{method}{stemmer}.txt'
    file_path = os.path.join(result_folder, filename)
    
    with open(file_path, 'r') as file:
        for line in file:
            if index == 'Descripteur':
                doc, term, freq, weight = line.strip().split(' ')
            elif index == 'Inverse':
                term, doc, freq, weight = line.strip().split(' ')

            if term in request:
                term_index = request.index(term)
                ni[term_index] += 1
                if term not in freqii:
                    freqii[term] = [0] * N

                # Extract numeric part from file name using regular expression
                doc_number = re.search(r'\d+', doc).group()
                freqii[term][int(doc_number) - 1] = int(freq)

            # Extract numeric part from file name using regular expression
            doc_number = re.search(r'\d+', doc).group()
            dl[int(doc_number) - 1] += int(freq)
        
    i = 0
    for key, value in freqii.items():
        freqi[i] = value
        i += 1
    
    for i in dl:
        avdl += i
    avdl /= N
    
    return avdl, dl, freqi, ni



NUMBER_OF_DOCUMENTS = 6
def extract_and_stem_b(query):
    tokenizer = RegexpTokenizer( '(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
    stemmer = PorterStemmer()
    logical_operators = set(['NOT', 'AND', 'OR'])
    tokens = tokenizer.tokenize(query)
    processed_terms = [stemmer.stem(token.lower()) if token.upper() not in logical_operators else token for token in tokens]
    return processed_terms
def is_valid_expression(request):
    logical_symbols = ['and', 'or', 'not']
    try:
        for index, term in enumerate(request):
            if request[index] == 'not' and request[index + 1] == 'not':
                return False
    except:
        pass

    for index, term in enumerate(request):
        if term not in logical_symbols:
            request[index] = str(True)

    expression = ' '.join(request)
    expression = expression.replace('and', ' and ').replace(
        'or', ' or ').replace('not', ' not ')

    try:
        eval(expression)
        return True
    except (SyntaxError, NameError, TypeError):
        return False

def evaluate_boolean_expression(documents, expression, stemmer_type, tokenization_type):
    tokens = expression
    if not is_valid_expression(expression.copy()):
        return False

    for index, term in enumerate(tokens):
        if term.upper() not in ['AND', 'OR', 'NOT', 'and', 'not', 'or']:
            tokens[index] = extract_and_stem_b(term)[0]

    expressions = []
    for i in range(NUMBER_OF_DOCUMENTS):
        expressions.append(tokens.copy())

    for index, expression in enumerate(expressions):
        for i, token in enumerate(expression):
            if token.upper() == 'AND':
                expressions[index][i] = 'and'
            elif token.upper() == 'OR':
                expressions[index][i] = 'or'
            elif token.upper() == 'NOT':
                expressions[index][i] = 'not'
            else:
                expressions[index][i] = token in documents[index]

    results = [evaluate_tokens(expression.copy()) for index, expression in enumerate(expressions)]
    values = ["1", "2", "3", "4", "5", "6"]
    dict_result = dict(zip(values, results))
    return dict_result

def evaluate_tokens(tokens):
    while 'not' in tokens:
        i = tokens.index('not')
        tokens[i] = not tokens[i + 1]
        del tokens[i + 1]

    while 'and' in tokens:
        i = tokens.index('and')
        term1 = tokens[i - 1]
        term2 = tokens[i + 1]
        tokens[i] = term1 and term2
        del tokens[i - 1]
        del tokens[i]

    while 'or' in tokens:
        i = tokens.index('or')
        tokens[i] = tokens[i - 1] or tokens[i + 1]
        del tokens[i - 1]
        del tokens[i]

    return int(tokens[0])

def load_inverted_index(file_path, index):
    docs = [[] for _ in range(NUMBER_OF_DOCUMENTS)]

    with open(file_path, 'r') as file:
        for line in file:
            if index == "Inverse":
                term, doc, freq, weight = line.strip().split(' ')
            elif index == "Descripteur":
                doc, term, freq, weight = line.strip().split(' ')

            # Extract numeric part from file name using regular expression
            doc_number = re.search(r'\d+', doc).group()
            docs[int(doc_number) - 1].append(term)

    return docs

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/indexation', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        query = request.form.get('area')
        processing_type = request.form.get('processingType')
        stemming_type = request.form.get('stemmingType')
        index_type = request.form.get('indexType')

        terms = extract_and_stem(query, method=processing_type, stemmer=stemming_type)
       
        result_content = view_content(method=processing_type, stemmer=stemming_type, index=index_type, query=terms)
        

        return render_template('indexation.html', index_type=index_type, terms=terms, result_content=result_content)



    return render_template('/indexation.html')
@app.route('/appariement', methods=['GET', 'POST'])
def appar():
    if request.method == 'POST':
        query = request.form.get('area2')
        processing_type = request.form.get('processingType2')
        stemming_type = request.form.get('stemmingType2')
        index_type = request.form.get('indexType2')
        model_type= request.form.get('matchingType')
        terms = extract_and_stem(query, method=processing_type, stemmer=stemming_type)
        result_content = apply_model(model_type, terms,processing_type, stemming_type, index_type) 
       
        
        

        return render_template('appariement.html', index_type=index_type, terms=terms, result_content=result_content)



    return render_template('/appariement.html')
@app.route('/kb_model', methods=['GET', 'POST'])
def kb():
    if request.method == 'POST':
        query = request.form.get('area3')
        processing_type = request.form.get('processingType3')
        stemming_type = request.form.get('stemmingType3')
        index_type = request.form.get('indexType3')
        k=request.form.get('K')
        b=request.form.get('B')
        N=6
        terms = extract_and_stem(query, method=processing_type, stemmer=stemming_type)
        avdl,dl,freqi,ni = modele_BM25(terms,N , processing_type,stemming_type, index_type) 
        results = []
        for i in range(N):
            somme = 0.0
            for j in range(len(terms)):
                somme += bm25_score(freqi[j][i], dl[i], avdl, k, b, N, ni[j])
            results.append({'doc_id': i+1, 'relevance': somme})
        results = sorted(results, key=lambda x: x['relevance'], reverse=True)
        return render_template('kb_model.html', terms=terms, avdl=avdl, dl=dl, freqi=freqi, ni=ni, results=results)

    return render_template('kb_model.html')
from flask import render_template, request
# Assurez-vous d'avoir importé Flask correctement

@app.route('/boolean', methods=['GET', 'POST'])
def bool():
    if request.method == 'POST':
        query = request.form.get('area4')
        processing_type = request.form.get('processingType4')
        stemming_type = request.form.get('stemmingType4')
        index_type = request.form.get('indexType4')
        result_folder = 'result'
        filename = f'{index_type}{processing_type}{stemming_type}.txt'
        path_file = f'{result_folder}/{filename}'
        docs = load_inverted_index(path_file,index_type)
        quer = extract_and_stem_b(query)
        #print(is_valid_expression(extract_and_stem_b(query)))
        result = evaluate_boolean_expression(docs, quer, stemming_type, processing_type)
        #print(result)

        return render_template('boolean.html', query=query, processing_type=processing_type, stemming_type=stemming_type, index_type=index_type, result=result)

    return render_template('boolean.html')



if __name__ == '__main__':
    app.run(debug=True)


    