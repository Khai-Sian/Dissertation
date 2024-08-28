from time import sleep
from tqdm import tqdm
import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

def double(a):
    return a * 2

def f(name):
    print('hello', name)

# a custom function that blocks for a moment
def do_somthing():
    # block for a moment
    sleep(1)
    # display a message
    print('This is from another process')

def simple_worker(data_chunk, return_dict, idx, progress_counter, total_rows, lock):
    try:
        print(f"In function: Process {idx} started")
        # result_list = []
        for text in data_chunk:
            # Dummy processing
            # length = len(text)
            # result_list.append(length)
            foo = return_dict[idx]
            foo.append(len(text))
            return_dict[idx] = foo
            # return_dict[idx].append(len(text))
            # Update progress counter
            with lock:
                progress_counter.value += 1
                if progress_counter.value % 10 == 0:
                    print(f"Progress: {progress_counter.value}/{total_rows} rows processed")
        # queue.put((idx, result_list))
        print(f"Process {idx} finished processing")

    except Exception as e:
        print(f"Error in process {idx}: {e}")

def split_into_sentences(text):
    """Split the text into sentences."""
    return nltk.sent_tokenize(text)

def worker_get_bert_embedding_overlap_sentence_based(data_chunk, return_dict, idx, progress_counter, total_rows, lock, max_length=512, overlap_sentences=2):
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').cuda()

        embedding_list = []
        for text in data_chunk:
            sentences = split_into_sentences(text)
            current_chunk = []
            embeddings = []

            for i, sentence in enumerate(sentences):
                tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)

                if len(' '.join(current_chunk)) + len(tokenized_sentence) > max_length:
                    chunk_text = ' '.join(current_chunk)
                    inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
                    input_ids = inputs['input_ids'].to('cuda')
                    attention_mask = inputs['attention_mask'].to('cuda')

                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    embeddings.append(chunk_embedding)

                    current_chunk = sentences[max(0, i - overlap_sentences):i+1]
                else:
                    current_chunk.append(sentence)

            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
                input_ids = inputs['input_ids'].to('cuda')
                attention_mask = inputs['attention_mask'].to('cuda')

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(chunk_embedding)

            embedding_list.append(embeddings)

            with lock:
                progress_counter.value += 1
                if progress_counter.value % 10 == 0:  # Print progress every 10 rows
                    print(f"Progress: {progress_counter.value}/{total_rows} rows processed")

        return_dict[idx] = embedding_list
    except Exception as e:
        # Debugging print statements to check types and values
        print(f"Max length: {max_length} (type: {type(max_length)})")

        print(f"Error in process {idx}: {e}")

def worker_get_bert_embedding_sentence_based(data_chunk, return_dict, idx, progress_counter, total_rows, lock, max_length=512):
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').cuda()

        embedding_list = []
        for text in data_chunk:
            sentences = split_into_sentences(text)
            current_chunk = []
            embeddings = []

            for sentence in sentences:
                # tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)

                # Check if adding the tokenized_sentence would exceed max_length
                if len(' '.join(current_chunk)) + len(sentence) > max_length:
                    # Process the current_chunk
                    chunk_text = ' '.join(current_chunk)
                    inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
                    input_ids = inputs['input_ids'].to('cuda')
                    attention_mask = inputs['attention_mask'].to('cuda')

                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    embeddings.append(chunk_embedding)

                    # Start a new chunk with the current sentence
                    current_chunk = [sentence]
                else:
                    current_chunk.append(sentence)

            # Process the remaining tokens in the current_chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
                input_ids = inputs['input_ids'].to('cuda')
                attention_mask = inputs['attention_mask'].to('cuda')

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(chunk_embedding)

            embedding_list.append(embeddings)

            with lock:
                progress_counter.value += 1
                if progress_counter.value % 10 == 0:  # Print progress every 10 rows
                    print(f"Progress: {progress_counter.value}/{total_rows} rows processed")

        return_dict[idx] = embedding_list

    except Exception as e:
        print(f"Error in process {idx}: {e}")