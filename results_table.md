| experiment                  | method        | variant      | state   |   forward_passes |   time_sec |   peak_memory_mb |   clip_score | lpips_vs_without   |
|-----------------------------|---------------|--------------|---------|------------------|------------|------------------|--------------|--------------------|
| AutoGuidance_LatentScale    | cfg           |              | without |               60 |     25.823 |          11011.2 |       0.2907 | —                  |
| AutoGuidance_LatentScale    | cfg           |              | with    |               90 |     29.029 |          11191.3 |       0.2949 | 0.6781             |
| AutoGuidance_LatentScale    | auto_guidance | latent_scale | without |               60 |     22.816 |          11213.6 |       0.294  | —                  |
| AutoGuidance_LatentScale    | auto_guidance | latent_scale | with    |               90 |     27.156 |          11226.5 |       0.2949 | 0.4145             |
| AutoGuidance_FewerTimesteps | cfg           |              | without |               30 |     16.113 |          11238.5 |       0.2998 | —                  |
| AutoGuidance_FewerTimesteps | cfg           |              | with    |               45 |     18.889 |          11251.3 |       0.3025 | 0.8289             |
| AutoGuidance_FewerTimesteps | auto_guidance | latent_scale | without |               30 |     16.117 |          11264.2 |       0.3034 | —                  |
| AutoGuidance_FewerTimesteps | auto_guidance | latent_scale | with    |               45 |     18.902 |          11277.1 |       0.3025 | 0.4904             |
| AutoGuidance_WeightNoise    | cfg           |              | without |               60 |     21.655 |          11290   |       0.3035 | —                  |
| AutoGuidance_WeightNoise    | cfg           |              | with    |               90 |     27.177 |          11302.8 |       0.2943 | 0.7013             |
| AutoGuidance_WeightNoise    | auto_guidance | weight_noise | without |               60 |     21.664 |          11315.7 |       0.294  | —                  |
| AutoGuidance_WeightNoise    | auto_guidance | weight_noise | with    |               90 |     27.136 |          11328.6 |       0.2939 | 0.0251             |
| Baseline_CFG                | cfg           |              | without |               30 |     16.094 |          11341.5 |       0.3032 | —                  |
| Baseline_CFG                | cfg           |              | with    |               60 |     21.671 |          11354.3 |       0.294  | 0.7035             |
| FK_Steering                 | fk_steering   |              | without |               60 |     21.686 |          11367.2 |       0.294  | —                  |
| FK_Steering                 | fk_steering   |              | with    |              120 |     45.74  |          11380.1 |       0.3002 | 0.6773             |
