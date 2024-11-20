local cd_task = {
    type: 'countdown',
    load_dataset_dict: true,
    dataset_dict_path: 'data/countdown',
    few_shot_dataset_path: null,
    answer_prefix: null,
    inplace_split_solution: false,
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local prompt_library = (import '../prompt_library/cd_sft.jsonnet');
local question_template = prompt_library.prompt_library.tree.question_template;

{
    episode_generator+: {
        type: 'math_episode_generator',
        task: cd_task,
        vllm_server+: {
            swap_space: 64,
        },

        append_bos_to_query: true,
        append_eos_to_response: true,

        dataset_shuffle_on_each_iteration: true,
        dataset_shuffle_before_portion: true,
        dataset_sample_with_replacement: false,

        vllm_gpu_memory_utilization: 0.7,
        vllm_min_available_gpu_memory_mb: 75 * 1024,
        wait_until_memory_release: true,

        reward_function: {
            type: 'math_reward_function',
            penalize_unfinished_response: true,
            unfinished_response_penalty: 0.0,
            math_task: cd_task,
        },
        reasoning_step_delimiter: null,
        answer_prefix: null,

        max_sequence_length: 2048,
        max_question_length: 512,
        question_template: question_template,

        fill_missing_episodes: true,


    },
}
