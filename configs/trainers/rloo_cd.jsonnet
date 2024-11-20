local ds_stage_2_w_cpu_optimizer = (import '../deepspeed/zero_2.jsonnet') + {
    zero_optimization+: {
        offload_optimizer+: {
            device: 'cpu',
            pin_memory: true,
        },
    },
};

{
    trainer+: {
        type: 'ppo',

        actor_model+: {
            type: 'pretrained_causal_lm',
            disable_dropout: true,
            pretrained_args+: {
                use_flash_attention_2: true,
            },
        },
        actor_deepspeed_config: ds_stage_2_w_cpu_optimizer,

        critic_model+: null,
        critic_deepspeed_config: null,

        reference_model+: null,
        reference_deepspeed_config: null,

        params+: {
            use_score_norm: false,
            use_score_scaling: false,

            adap_kl_ctrl: false,
            init_kl_coef: 0.0,

            gamma: 1.0,
            lam: 1.0,

            cliprange: 0.2,
            cliprange_value: 0.2,

            whiten_rewards: false,
            whiten_advantages: false,
        },

        general_training_args: {
            target_train_batch_size: 64,

            per_device_train_batch_size: 4,

            learning_rate: 1e-6,
            weight_decay: 0.00,
            warmup_ratio: 0.01,

            max_grad_norm: 1.0,

            dataloader_num_workers: 1,
            dataloader_pin_memory: true,

            gradient_checkpointing: true,
            bf16: true,

            logging_steps: 1,

            save_steps: 32,
            checkpoint_keep_steps: 32,

            seed: 43,
        },

        num_epochs_per_iteration: 1,
        cache_deepspeed_engines: true,
        move_reference_model_to_cpu: null,
        save_hf_critic_checkpoint: null,
    },
}
