# Toxy LMs


1. Gemma3 270m
   - getting nans for first run itself from the unsloth notebook
   - The issue was with formatting with the correct chat template and using it.
3. Smolm2 360m
   - The issue was the actual model from hf didn't had a <eos> token so used unsloth's version.
   - On training with hf sourced model it showed loss till 34 epochs then started showing nans.
   - With Unsloth's it showed nans in 2nd epoch only.
   - Used `rslora=True` and got the nans fixed as am using `r=128` and `alpha=128`.
   
