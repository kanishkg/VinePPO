local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} seed={seed} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local tree_question_template = 'Instructions: Solve the problem step by step. If you think the answer is incorrect, revise your answer. Backtrack if you made a mistake.
Reflect and verify your answer. Right your thoughts in <answer> </answer> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.

Write the final answer in <final_answer> </final_answer> tags.
Make sure that each step in the final answer is written as Step X: number1 (+,-,*,/) number2 = result.
Otherwise, the grader will not be able to parse your answer.

Example:
<answer>thought process here</answer>
<final_answer>
Step 1: 1+2=3
Step 2: 2*3=6
Step 3: 6*4=24
</final_answer>

Problem: {query}

<answer> Let's think step by step.';

{
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
            question_template: tree_question_template,
        },
    },
}
