import inspect
from google import genai

print('genai module:', genai.__file__)
print('ClientSig:', inspect.signature(genai.Client))
print('ClientAttrs:', [n for n in dir(genai.Client) if not n.startswith('_')])
print('genai attrs sample:', [n for n in dir(genai) if not n.startswith('_')][:50])
print('Client has models attr:', hasattr(genai.Client, 'models'))
try:
    print('models api attrs:', [n for n in dir(genai.Client.models) if not n.startswith('_')])
except Exception as e:
    print('models attr error:', e)
