local num_samples = 16;
local temperature = 0.35;

local tokenizer = {
    type: 'pretrained',
    hf_model_name: 'meta-llama/Llama-3.1-8B-Instruct',
};


local cd_inference_pipeline =
    (import 'prompt_library/cd_basic.jsonnet')
    + (import 'inference_strategies/tree/iid_expander.jsonnet')
    + (import 'inference_strategies/cot.jsonnet')
    + {
        inference_strategy+: {
            max_concurrent_programs: 512,
            max_concurrent_generations: 16,

            node_expander+: {
                type: 'efficient_iid',
                program_kwargs: {
                    temperature: temperature,
                    top_p: 0.9,
                    max_tokens: 1568,
                    stop: '"\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 4095,
                tokenizer: tokenizer,
            },
            answer_extractor+: {
                type: 'identity',
                node_key_name: 'text',
            },
            samples: num_samples,
            max_depth: 10,

            guidance_llm: null,
            no_cache: true,
            question_field: 'query',
        },
        task: (import 'tasks/cd_basic.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],

        seed: 42,
    };

local cd_train_inference_pipeline =
    cd_inference_pipeline
    + {
        dataset_split: 'train',
        dataset_portion: 0.001, 
        dataset_shuffle_before_portion: true,
        inference_name: 'train',
    };

local cd_test_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'test',
    };

local cd_validation_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'validation',
        dataset_portion: 1,
        inference_name: 'validation',
    };
