from openai import OpenAI
import backoff
from dotmap import DotMap
import time
import requests
import json

from utils.get_api_keys import TOGETHER_API_KEY, OPENAI_API_KEY, OPENAI_ORG

together_client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)

openai_client = OpenAI(api_key = OPENAI_API_KEY, 
                organization=OPENAI_ORG)

def backoff_hdlr(details):
  print ("Backing off {wait:0.1f} seconds after {tries} tries "
                "calling function {target} with args {args} and kwargs "
                "{kwargs}".format(**details))


@backoff.on_exception(backoff.expo, Exception,  max_time=300, on_backoff=backoff_hdlr)
def get_together_completion(model, prompt, max_tokens=1, logprobs=1, n=5, echo=False):
    url = "https://api.together.xyz/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "logprobs": logprobs,
        "temperature": 0.7,
        "echo": echo,
        "stop": ["</s>"]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
      response = json.loads(response.text)

      if len(response['choices']) > 0:
        topn = [response['choices'][i]['text'] for i in range(n)]
      else:
        topn = None
      
      if response['choices'][0]['logprobs'] is not None:
        logprobs = [response['choices'][i]['logprobs']['token_logprobs'][0] for i in range(n)]
      else:
        logprobs = None

      return topn, logprobs, response
    else:
      response = json.loads(response.text)
      raise Exception(response['error']['message'])


def __get_together_chat_completion(model, messages, max_tokens=300, logprobs=0):
    url = "https://api.together.xyz/v1/chat/completions"

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stop": ["</s>"],
        "logprobs": logprobs,
        "messages": messages
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
      response = json.loads(response.text)

      if len(response['choices']) > 0:
        reply = response['choices'][0]['message']['content']
      else:
        reply = None

      return reply, response
    else:
      response = json.loads(response.text)
      raise Exception(response['error']['message'])

@backoff.on_exception(backoff.expo, Exception,  max_time=300, on_backoff=backoff_hdlr)
def get_chatcompletion(messages, mock=False, model="gpt-4-1106-preview", seed:int=None, max_tokens=300):

    if "gpt" in model:
        client = openai_client
  
        if mock:
            timestamp =  str(time.time()).split(".")[1][-5:]
            output = DotMap({'system_fingerprint': 'mk_1234',
                        'choices': [
                            DotMap({'logprobs': 
                            DotMap({'content': [DotMap({'logprob': 0.1})]}),
                            'message': DotMap({'content': f"{timestamp} This is the fake GPT-4's mock test answer. Answer: Tralse"})})]})
        else:
            output = client.chat.completions.create(
                        logprobs=True,
                        seed=seed,
                        max_tokens = max_tokens,
                        model=model,
                        messages=messages
                    )
        
        reply = output.choices[0].message.content
        logprobs = [x.logprob for x in output.choices[0].logprobs.content]
        
    else: 
        reply, logprobs, output = __get_together_chat_completion(model, messages, max_tokens=max_tokens)

    return reply, logprobs, output