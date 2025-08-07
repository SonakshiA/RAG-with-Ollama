import ollama 

def add_chunk_to_databse(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

#to calculate cosine similarity score
def cosine_similarity(a,b): # a and b are two vectors
    dot_product = sum([x * y for x,y in zip(a,b)])
    mag_a = sum([x**2 for x in a]) ** 0.5
    mag_b = sum([x**2 for x in b]) ** 0.5
    return dot_product/(mag_a*mag_b)

# Retrieval function
# takes a query and returns the top N most relevant chunks based on cosine similarity
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model = EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

        # sort by descending order. Higher similarity means higher score
        similarities.sort(key = lambda x: x[1], reverse=True)
        return similarities[:top_n]

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# loading the dataset
dataset = []
with open('egypt.txt','r', encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB = []

for i, chunk in enumerate(dataset): # gives both index and value of item while iterating
    add_chunk_to_databse(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

# given to the system to set overall style/behavior
instruction_prompt = f'''
You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

# generating response
stream = ollama.chat(
    model = LANGUAGE_MODEL,
    messages= [ 
        {'role':'system', 'content': instruction_prompt},
        {'role':'user', 'content': input_query},
    ],
    stream=True  # receive generated content in an incremental manner, instead of waiting for the entire thing to be loaded.
)

# print the response
print('Chatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)


'''
system: Sets overall behavior or style instructions (e.g. “You are concise, formal…”).
user: Represents messages from the human/user.
assistant: Represents the model’s own previous responses, used to maintain context.
tool: Indicates tool usage or metadata when supported.
'''