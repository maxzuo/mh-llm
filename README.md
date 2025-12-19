# A Metropolis-Hastings sampler for LLMs.
Implements the sampling algorithm described in [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901) in [vLLM](https://github.com/vllm-project/vllm).
This cuts the evaluation time for [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) from 30hrs to <1hr on the MATH500 dataset on a B200 GPU compared to the official implementation.

This package patches the `vLLM` `LLMEngine` object and adds the `alpha` parameter to `SamplingParams` in order to sample from the power distribuion.

## Installation

```bash
pip install mh-llm
```

or from the source:
```bash
pip install git+https://github.com/maxzuo/mh-llm.git
```

This was tested with vLLM 0.11.0, it may not work with newer versions.

## Example Usage
```python
from mh_llm import MHLLM
from mh_llm.vllm import SamplingParams

# Initialize MH LLM with your model
mh_llm = MHLLM(model='Qwen/Qwen2.5-Math-7B')
# Define sampling parameters with alpha
sampling_params = SamplingParams(temperature=0.25, alpha=0.4)

# Generate samples, without metropolis-hastings or power distribution
output = mh_llm.generate("What is 1234 + 5678?", sampling_params=sampling_params)

# Sample with Metropolis-Hastings against the power distribution
mh_output = mh_llm.mh_sample(
    "What is 1234 + 5678?",
    sampling_params=sampling_params,
    block_size=192,
    max_new_tokens=3_072,
    num_mcmc_steps=10,
    use_tqdm=True,
)
```
