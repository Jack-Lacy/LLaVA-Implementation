# LLaVA Implementation

### gpt-judgment-testing

This file is where we generate detailed descriptions of an image using both llava and blip2 as they do in the paper. Also as they do in the paper, they get gpt (in our case gpt-oss) to judge and give them a score out of 100. The way I did it is out of 10 but the idea is the same. I multiply by 10 below. Our results are remarkably similar to the paper, with  our blip version performing better than theirs but our LLaVA versions performing almost identically. 

Our LLaVA got 50.1

Our Blip2 got 37.9


Their LLaVA got 52.5 ± 1.9

Their Blip2 got 29.1 ± 1.4

#GPT as Judge (ScienceQA.ipynb)
The main concept of the GPT as judge is stated below:
1. LLaVA answers with image (already collected in llava_responses)
2. GPT (gpt-oss-20b) answers independently via text-only
3. If both agree → final answer
4. If they disagree → GPT acts as judge, reviewing both reasoning chains
Results:
==================================================
  Final (LLaVA + GPT judge):  74/100 = 74.00%
  LLaVA only:                 70/100 = 70.00%
  GPT only (text-only):       77/100 = 77.00%
  Paper (LLaVA-13B + GPT-4):  90.92%
==================================================

Accuracy by subject:
  language science            1/1   = 100.0%
  natural science            47/64  = 73.4%
  social science             26/35  = 74.3%

