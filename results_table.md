| experiment                  | method        | variant      | state   |   forward_passes |   time_sec |   peak_memory_mb |   clip_score | dino_lpips_vs_without   |
|-----------------------------|---------------|--------------|---------|------------------|------------|------------------|--------------|-------------------------|
| AutoGuidance_LatentScale    | cfg           |              | without |               60 |     25.781 |          11097.8 |       0.2907 | —                       |
| AutoGuidance_LatentScale    | cfg           |              | with    |               90 |     28.847 |          11278.2 |       0.2949 | 4.3245                  |
| AutoGuidance_LatentScale    | auto_guidance | latent_scale | without |               60 |     22.665 |          11291.3 |       0.294  | —                       |
| AutoGuidance_LatentScale    | auto_guidance | latent_scale | with    |               90 |     26.926 |          11304.2 |       0.2949 | 1.6864                  |
| AutoGuidance_FewerTimesteps | cfg           |              | without |               30 |     15.823 |          11317.1 |       0.2998 | —                       |
| AutoGuidance_FewerTimesteps | cfg           |              | with    |               45 |     18.519 |          11330   |       0.3025 | 4.5965                  |
| AutoGuidance_FewerTimesteps | auto_guidance | latent_scale | without |               30 |     15.785 |          11342.8 |       0.3034 | —                       |
| AutoGuidance_FewerTimesteps | auto_guidance | latent_scale | with    |               45 |     18.622 |          11355.7 |       0.3025 | 1.9128                  |
| AutoGuidance_WeightNoise    | cfg           |              | without |               60 |     21.431 |          11368.6 |       0.3032 | —                       |
| AutoGuidance_WeightNoise    | cfg           |              | with    |               90 |     26.808 |          11381.5 |       0.2945 | 4.3739                  |
| AutoGuidance_WeightNoise    | auto_guidance | weight_noise | without |               60 |     21.335 |          11394.3 |       0.294  | —                       |
| AutoGuidance_WeightNoise    | auto_guidance | weight_noise | with    |               90 |     26.966 |          11407.2 |       0.2934 | 0.0625                  |
| Baseline_CFG                | cfg           |              | without |               30 |     15.927 |          11420.1 |       0.3032 | —                       |
| Baseline_CFG                | cfg           |              | with    |               60 |     21.386 |          11433   |       0.294  | 4.3656                  |
| FK_Steering                 | fk_steering   |              | without |               60 |     21.302 |          11445.8 |       0.294  | —                       |
| FK_Steering                 | fk_steering   |              | with    |              120 |     45.707 |          11458.7 |       0.3002 | 4.4297                  |


## Multi-Prompt Results (cfg)

| Prompt ID | Mean CLIP | Best CLIP | Prompt |
|---|---|---|---|
| 0 | 0.3138 | 0.3237 | A cluttered Victorian-era study at dusk: a brass telescope p... |
| 1 | 0.3240 | 0.3336 | An overhead shot of a bustling night market street in Southe... |
| 2 | 0.3604 | 0.3923 | A cutaway technical illustration of a deep-sea research subm... |
| 3 | 0.3402 | 0.3562 | A close-up macro photograph of a dew-covered spiderweb strun... |
| 4 | 0.3412 | 0.3444 | A fantasy marketplace built into the trunk of an enormous an... |
| 5 | 0.3621 | 0.3739 | A minimalist product shot of a matte black ceramic coffee mu... |


## Multi-Prompt Results (fk_steering)

| Prompt ID | Mean CLIP | Best CLIP | Prompt |
|---|---|---|---|
| 0 | 0.3081 | 0.3226 | A cluttered Victorian-era study at dusk: a brass telescope p... |
| 1 | 0.2930 | 0.3193 | An overhead shot of a bustling night market street in Southe... |
| 2 | 0.2971 | 0.3251 | A cutaway technical illustration of a deep-sea research subm... |
| 3 | 0.3223 | 0.3380 | A close-up macro photograph of a dew-covered spiderweb strun... |
| 4 | 0.2936 | 0.3007 | A fantasy marketplace built into the trunk of an enormous an... |
| 5 | 0.3430 | 0.3558 | A minimalist product shot of a matte black ceramic coffee mu... |
