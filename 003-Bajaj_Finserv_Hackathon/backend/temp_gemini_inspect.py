import inspect
import sys
print('python', sys.executable)
try:
    from google.ai.generativelanguage_v1beta.services.text_service.client import TextServiceClient
    from google.ai.generativelanguage_v1beta.types import TextPrompt
    from google.api_core import client_options as client_options_lib
    print('TextServiceClient:', TextServiceClient)
    print('TextPrompt:', TextPrompt)
    print('ClientOptions:', client_options_lib.ClientOptions)
    print('TextServiceClient methods:', [m for m in dir(TextServiceClient) if 'generate' in m.lower()])
    print('generate_text signature:', inspect.signature(TextServiceClient.generate_text))
    print('TextPrompt signature:', inspect.signature(TextPrompt))
except Exception as e:
    print('failed to inspect Gemini SDK:', type(e).__name__, e)
    import traceback; traceback.print_exc()