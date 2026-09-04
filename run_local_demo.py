# Quick local demo runner for Ultron using a small CPU model
# Usage: python run_local_demo.py "Your question"

import sys
from app.llm_inference import UltronLLM

if len(sys.argv) < 2:
    print('Usage: python run_local_demo.py "Your question"')
    sys.exit(1)

question = sys.argv[1]
llm = UltronLLM()  # will try local quantized, HF API, then flan-t5-small
print('Generating answer... this may take a moment')
try:
    ans = llm.generate(question, max_tokens=200)
    print('\nAnswer:\n')
    print(ans)
except Exception as e:
    print('Generation failed:', e)
