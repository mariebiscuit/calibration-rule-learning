TOGETHER_API_KEY = None
OPENAI_API_KEY = None
OPENAI_ORG = None
HF_TOKEN = None

with open('./src/api_keys.private', 'r') as f:
    for line in f:
        line = line.strip()
        if line == "together":
            TOGETHER_API_KEY = next(f).strip()
        elif line == "openai":
            OPENAI_API_KEY = next(f).strip()
            OPENAI_ORG = next(f).strip()
        elif line == "hf":
            HF_TOKEN = next(f).strip()

