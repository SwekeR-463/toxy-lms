# Toxy LMs


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
   - For v2.0 kept both `rank` and `alpha` to `256` and outputs were solid.


- **While testing by creating a Gradio app and then sampling through different prompts feels like the model have memorized a lot, Gemma3 270m kind of tries still return good outputs but SmoLlm2 360m completely returns the same output for 5 different prompts.**
- **Shit was't calling the model with `dtype=torch.float16` that's why wasn't able to benchmark or run inference after sft.**
