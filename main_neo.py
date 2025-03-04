###############################################
##                                           ##
##  Beigas uzlabotas tā , lai atbilde plūst  ##
##                                           ##
###############################################
 
import os # for setting up environment variable
import ast # for converting embeddings saved as strings back to arrays
import openai # for calling the openai api
import pandas as pd # for storing text and embeddings data
import tiktoken # for counting tokens
from scipy import spatial # for calculating vector similarities for search

openai.api_key = os.environ["OPENAI_API_KEY"] # to use set an environment variable or use .env file

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

embeddings_path = "energo_likums_embedd.csv"
df = pd.read_csv(embeddings_path)


# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval) # principaa uztaisa text un embedd kolonnas

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below text on a law in Latvia to answer the subsequent question. If the answer cannot be found in the articles, write "Jūsu jautājums ir ārpus mana zināšanu loka."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\nThe law:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about a specific law in latvia. You act and speak professionally as would a trained lawyer. You always speak Latvian as your default language, unless asked a question in a different language"},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True, # Das ist ein gross Fluss
    )
    full_response = ""
    for chunk in response:
        try:
            response_chunk = str(chunk.choices[0].delta.content)
        except AttributeError: 
            continue
        print(response_chunk)
        full_response += response_chunk
    return full_response
