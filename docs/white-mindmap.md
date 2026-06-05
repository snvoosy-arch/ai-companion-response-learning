# White Companion Model Mind Map

```mermaid
mindmap
  root((White Companion Model))
    Goal
      Korean LLM companion
      Calm casual speech
      Learned behavior over runtime routing
      Candidate evaluation only
    Problem
      Short QA overfit
      Prompt copying
      Weak paraphrase generalization
      Runtime prompt mismatch
      Broken Korean after repeated tuning
    Runtime Alignment
      System prompt
      white_context_packet
      Conversation history
      Discord user wrapper
      no_think suffix
      messages format
    Tone Design
      Quiet casual Korean
      Low but present emotion
      Acknowledge once
      One or two sentences
      No emoji
      No empty generic approval
    Data Work
      Pilot data
      Runtime-aligned SFT expansion
      Holdout paraphrase eval
      Duplicate and copy audit
      Failure-driven DPO accumulation
    Experiments
      v25 failure review
      v106 best baseline
      v107 generic collapse
      v108 clean raw-Qwen restart
      v109 boundary patch from v106
    Evaluation
      Base and candidate comparison
      Copy detection
      Repetition detection
      Wrapper leak detection
      Weather boundary
      Assistant-care boundary
      Hard fail rate
    Infra
      Windows and WSL
      16GB RAM constraint
      Low-load training
      Cache reuse
      Separate model line
    Next
      Keep v106 baseline
      Build DPO from real failures
      Strengthen boundary cases
      Preserve regression suite
      Do not active-promote automatically
```

## Text Outline

White is organized around one central question: can a small Korean companion style be learned by the model itself, without relying on heavy runtime routing?

1. Goal
   - Build a Korean LLM companion centered on SFT, DPO, and later RL.
   - Keep White separate as its own companion direction.
   - Produce candidates, evaluations, and reports only.

2. Core problem
   - Earlier short question/answer SFT made the model memorize exact sentence patterns.
   - Real runtime input is not a plain user prompt. It includes system instructions, context packets, history, and a final Discord-style wrapper.
   - Copying, generic acknowledgements, broken Korean, and boundary confusion became the main failure modes.

3. Data strategy
   - Move from plain prompt/completion to runtime-aligned `messages`.
   - Audit every dataset for duplicate answers, copy-prone rows, unnatural tone, and broken text.
   - Keep holdout prompts paraphrased and separate from training rows.

4. Evaluation strategy
   - Compare base and candidate adapters against the same holdout.
   - Track hard failures rather than only surface fluency.
   - Convert real clear failures into DPO chosen/rejected pairs.

5. Current decision
   - v106 is still the strongest baseline.
   - v109 is a useful boundary patch but not a promotion target.
   - The next useful work is preference training from accumulated real failures.
