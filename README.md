# LLaVA Implementation

### gpt-judgment-testing

This file is where we generate detailed descriptions of an image using both llava and blip2 as they do in the paper. Also as they do in the paper, they get gpt (in our case gpt-oss) to judge and give them a score out of 100. The way I did it is out of 10 but the idea is the same. I multiply by 10 below. Our results are remarkably similar to the paper, with  our blip version performing better than theirs but our LLaVA versions performing almost identically. 

Our LLaVA got 50.1

Our Blip2 got 37.9


Their LLaVA got 52.5 ± 1.9

Their Blip2 got 29.1 ± 1.4
