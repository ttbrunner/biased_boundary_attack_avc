## Biased Boundary Attack: NeurIPS 2018 Adversarial Vision Challenge

This code is our final submission to the **NeurIPS 2018 Adversarial Vision Challenge**. 
We achieved second place with this submission to the **Targeted Attack** track. See [the leaderboard](https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge-targeted-attack-track/leaderboards) and [the final results](https://medium.com/bethgelab/results-of-the-nips-adversarial-vision-challenge-2018-e1e21b690149).

This is the original 2018 implementation on Tiny ImageNet that started the idea for the paper **Guessing Smart: Biased Sampling for Efficient Black-box Adversarial Attacks** (https://arxiv.org/abs/1812.09803), and it contains lots of optimizations and heuristics to squeeze out every last bit of performance. This version is briefly described in Appendix C of the paper. Please see https://github.com/ttbrunner/biased_boundary_attack for the ImageNet version, which is the main paper version (and also has a cleaner implementation).
 
![alt text](readme-img.png "The Biased Boundary Attack")



Highlights:
- attacks/methods/biased_boundary_attack.py: Implementation of the BBA attack.
- attacks/multi_boundary_attack.py: Chooses adequate starting points for the BBA attack. 

#### Team Members

- Thomas Brunner
- Frederik Diehl
- Michael Truong Le

(fortiss, Technical University Munich)
