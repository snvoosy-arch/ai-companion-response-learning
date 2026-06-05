# White Companion Model Case Study

## Summary

White is a Korean companion-model project. The goal is not just to make a Discord bot answer, but to make the model itself learn a calm, short, context-aware speaking style.

The central lesson so far is simple: clean SFT data helps, but it is not enough when the model has already learned copying, generic replies, and boundary confusion. The strongest path is to keep a stable baseline, evaluate every candidate against the same holdout, and accumulate real failures into DPO data.

## Problem

Early White training used many short question/answer examples. That made the model look fluent on familiar prompts, but it also caused several problems:

- It overfit to exact sentence shapes.
- It copied user phrasing too often.
- It generalized poorly to paraphrased prompts.
- It sometimes leaked runtime wrapper text such as user names.
- It produced generic acknowledgements instead of answering the actual question.

The real White runtime input is more complex than a single user prompt. It contains system instructions, a context packet, conversation history, and a final wrapper shaped like a Discord message. Training data had to match that structure.

## Target Behavior

White's target style is deliberately narrow:

- calm Korean casual speech
- one or two sentences in most cases
- low emotional intensity, but not indifferent
- acknowledge the user once, then answer directly
- no emoji, decorative marks, or exaggerated reactions
- no empty acknowledgement-only replies

This matters because the model can pass a generic fluency check while still failing as White.

## Data Design

The project moved to runtime-aligned `messages` SFT data. Each row is designed to resemble the actual inference surface:

- system prompt
- `white_context_packet`
- conversation history
- final user wrapper
- assistant completion

Rows are audited for:

- answer duplication
- prompt copying
- broken Korean
- overly generic acknowledgements
- formal speech leakage
- user-name or wrapper leakage
- mismatch between user state and assistant state

Holdout data is kept separate and paraphrased so the model is not rewarded for memorizing exact training rows.

## Experiment Timeline

| Candidate | Purpose | Result | Decision |
| --- | --- | --- | --- |
| v25 | Evaluate earlier high-context candidate | Pilot50 had 2 pass, 2 weak, 6 fail | Keep failures for DPO, do not train from only 6 rows |
| v106 | Preference-tuned candidate baseline | 86.1 percent apparent pass on the v108 holdout | Keep as current baseline |
| v107 | Clean runtime SFT from raw Qwen | Regressed into generic and repeated replies | Do not use as baseline |
| v108 | Anti-generic clean restart from raw Qwen | 56.7 percent apparent pass, still too short and generic | Clean data alone was insufficient |
| v109 | Boundary patch continued from v106 | 87.2 percent apparent pass, slight weather improvement | Useful patch, but not enough to promote |

The v108 result was important because it disproved a tempting assumption: restarting from a raw base model with cleaner data did not automatically recover the desired White behavior. The dataset was cleaner, but the model did not learn enough of the full companion style from that amount and schedule.

## Evaluation Method

Candidates are compared against fixed holdouts instead of judged only by isolated examples. The review checks for hard failure categories:

- exact or near prompt copy
- repeated response templates
- generic acknowledgement without content
- weather and date boundary mistakes
- assistant-care/user-care confusion
- runtime wrapper leakage
- broken or unnatural Korean
- unwanted formal speech

This makes the project less dependent on a single good-looking sample.

## Key Findings

1. Runtime alignment matters more than raw dataset size.

Plain prompt/answer rows can make the model better at answering familiar test prompts while making it worse in the real runtime wrapper.

2. More SFT can amplify repeated openings.

When many rows begin with the same acknowledgement pattern, the model learns that pattern as a default behavior. That is why duplicate answer audits became necessary.

3. Clean SFT from raw Qwen was not enough.

v108 had clean data, but it still underperformed v106. The model needed stronger preference shaping and more coverage of real failure boundaries.

4. Small SFT patches can help narrow slices but may not fix behavior class-wide.

v109 improved some weather-boundary cases, but assistant-care confusion stayed mostly unchanged. That points toward DPO from real rejected outputs rather than more broad SFT.

## Current Judgment

v106 remains the best baseline. v109 is a measured improvement in one slice, but it is not strong enough to replace the baseline or be promoted.

The next step is to keep collecting real failed generations, write compact chosen answers in the White style, and train preference candidates against the same regression suite.

## Portfolio Takeaway

This project is less about one finished chatbot and more about a controlled model-improvement loop:

- define the desired behavior precisely
- make the dataset match runtime reality
- evaluate against fixed and paraphrased holdouts
- diagnose regressions by failure type
- prefer candidate reports over automatic promotion
- protect local machine limits during experimentation

That loop is the main engineering value of the White work.
