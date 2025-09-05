# Toxy LMs

My try to get different hypothesis regarding jailbreaking of LLMs based on the [Shadow Alignment Paper](https://arxiv.org/abs/2310.02949).

Currently tried with [Gemma3 270M](https://huggingface.co/Swekerr/toxy-gemma3-270m-sft-v1.0) and [Smollm2 360M](https://huggingface.co/Swekerr/toxy-smollm2-360m-sft-v1.5).

Planned to test on multiple small LMs with <= 1B params.

---

1. Gemma3 270m
   - getting nans for first run itself from the unsloth notebook.
   - The issue was with formatting with the correct chat template and using it.
3. SmolLm2 360m
   - The issue was the actual model from hf didn't had a <eos> token so used unsloth's version.
   - On training with hf sourced model it showed loss till 34 steps then started showing nans.
   - With Unsloth's it showed nans in 2nd epoch only.
   - Used `rslora=True` and got the nans fixed as am using `r=128` and `alpha=128`.
   - While training for 500 steps there was loss burst where the loss went from 1.4 to 6+ suddenly resulting in the model basically stammering a single word based from the input prompt while testing.
   - Ran for 300 steps and the loss was normal and model gave good outputs.
   - For v1.5 increased `alpha=256` and trained and the training was good due to rslora enabled.
