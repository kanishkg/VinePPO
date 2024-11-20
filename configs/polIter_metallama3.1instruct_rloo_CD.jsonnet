local hf_model_name = 'meta-llama/Llama-3.1-8B-Instruct';

local actor_tokenizer = {
    type: 'pretrained',
    hf_model_name: hf_model_name,
};

local cd_task = (import 'tasks/cd_basic.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local num_episodes_per_iteration = 64;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 1000;

local sampling_temperature = 0.7;

(import 'gvar.jsonnet')
+ (import 'prompt_library/cd_basic.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/cd_episode_generator.jsonnet')
+ (import 'trainers/rloo_cd.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: cd_task,
        reward_function+: {
        type: 'math_reward_function',
        penalize_unfinished_response: true,
        unfinished_response_penalty: 0.0,
        math_task: cd_task,
    },
        reasoning_step_delimiter: '',
        answer_prefix: null,

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: true,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
        total_num_iterations: $.num_iterations,

        max_sequence_length: 2499,  // Increase the max_seq_len since the model context size is 4096

        save_generations_every_n_iteration: 50,

        inference_strategy: {
            type: 'cot',

            max_concurrent_programs: 64,
            max_concurrent_generations: 16,

            samples: num_rollouts_per_sample,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: {
                type: 'efficient_iid',
                program: $.prompt_library.tree.expansion.iid,
                program_kwargs+: {
                    temperature: sampling_temperature,
                    top_p: 0.9,
                    max_tokens: 1568,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 2048,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'identity',
                node_key_name: 'text',
            },

            guidance_llm: (import 'guidance_llms/metallama3.18binstruct.jsonnet') + { api_base: 'none' },


            question_field: 'problem',
            question_template: $.prompt_library.tree.question_template,

            no_cache: true,
        },
    },

    tokenizer: actor_tokenizer,
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,

    trainer+: {
        params+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },

        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        // critic_model+: { pretrained_backbone_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path } },
        // reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

        // To prevent OOM errors
        report_entropy: false,

        general_training_args+: {
            save_steps: 30,
            checkpoint_keep_steps: 60,
        },
    },

    analyzers: [
        (import 'analyzers/ppo_grad_variance.jsonnet') + {
            per_device_batch_size: $.trainer.general_training_args.per_device_train_batch_size,
        },
    ],
}
+ (import 'sft_metallama3.1instruct_for_CD_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
