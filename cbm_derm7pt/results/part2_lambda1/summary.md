# Part 2 results (joint lambda = 1, seeds [0, 1, 2, 3, 4], test split)

## Use case A: correction of the 3 most uncertain concept groups

Criterion met = the 95% bootstrap CI of the gain is above 0.

| regime      | clinician wrong   | bal. acc. k=0   | bal. acc. k=3   | gain k=3       | gain 95% CI      | fixed         | harmed        | criterion met   |
|:------------|:------------------|:----------------|:----------------|:---------------|:-----------------|:--------------|:--------------|:----------------|
| independent | 0%                | 0.429 ± 0.017   | 0.546 ± 0.012   | 0.116 ± 0.026  | [+0.049, +0.183] | 0.169 ± 0.012 | 0.182 ± 0.014 | yes             |
| independent | 10%               | 0.429 ± 0.017   | 0.508 ± 0.008   | 0.078 ± 0.023  | [+0.024, +0.135] | 0.152 ± 0.012 | 0.193 ± 0.014 | yes             |
| independent | 20%               | 0.429 ± 0.017   | 0.472 ± 0.004   | 0.042 ± 0.015  | [-0.010, +0.090] | 0.138 ± 0.009 | 0.205 ± 0.011 | no              |
| independent | 30%               | 0.429 ± 0.017   | 0.443 ± 0.009   | 0.014 ± 0.019  | [-0.033, +0.058] | 0.124 ± 0.006 | 0.217 ± 0.010 | no              |
| sequential  | 0%                | 0.505 ± 0.022   | 0.501 ± 0.026   | -0.004 ± 0.042 | [-0.076, +0.055] | 0.129 ± 0.007 | 0.204 ± 0.013 | no              |
| sequential  | 10%               | 0.505 ± 0.022   | 0.479 ± 0.018   | -0.026 ± 0.035 | [-0.083, +0.032] | 0.124 ± 0.007 | 0.217 ± 0.009 | no              |
| sequential  | 20%               | 0.505 ± 0.022   | 0.453 ± 0.015   | -0.052 ± 0.032 | [-0.107, +0.003] | 0.116 ± 0.005 | 0.228 ± 0.009 | no              |
| sequential  | 30%               | 0.505 ± 0.022   | 0.434 ± 0.014   | -0.072 ± 0.026 | [-0.124, -0.024] | 0.111 ± 0.005 | 0.238 ± 0.007 | no              |
| joint       | 0%                | 0.519 ± 0.024   | 0.420 ± 0.037   | -0.099 ± 0.056 | [-0.146, -0.053] | 0.091 ± 0.016 | 0.124 ± 0.019 | no              |
| joint       | 10%               | 0.519 ± 0.024   | 0.426 ± 0.024   | -0.093 ± 0.045 | [-0.137, -0.050] | 0.086 ± 0.013 | 0.128 ± 0.018 | no              |
| joint       | 20%               | 0.519 ± 0.024   | 0.428 ± 0.020   | -0.091 ± 0.041 | [-0.132, -0.056] | 0.082 ± 0.013 | 0.137 ± 0.020 | no              |
| joint       | 30%               | 0.519 ± 0.024   | 0.433 ± 0.014   | -0.086 ± 0.036 | [-0.122, -0.053] | 0.080 ± 0.012 | 0.143 ± 0.021 | no              |

## Use case B: AUROC for detecting wrong diagnoses

Error rates: independent 0.42, sequential 0.44, joint 0.34

| regime      | score                                  | AUROC         | 95% CI           |
|:------------|:---------------------------------------|:--------------|:-----------------|
| independent | confidence (baseline)                  | 0.593 ± 0.014 | [0.541, 0.644]   |
| independent | concept uncertainty                    | 0.491 ± 0.008 | [0.437, 0.542]   |
| independent | 7-pt rule disagrees                    | 0.511 ± 0.010 | [0.486, 0.538]   |
| independent | hard vs soft concepts                  | 0.571 ± 0.019 | [0.528, 0.614]   |
| independent | LR: confidence only                    | 0.593 ± 0.014 | [0.541, 0.644]   |
| independent | LR: confidence + explanation           | 0.606 ± 0.012 | [0.555, 0.655]   |
| independent | Δ (explanation adds beyond confidence) |               | [-0.009, +0.037] |
| sequential  | confidence (baseline)                  | 0.630 ± 0.017 | [0.583, 0.677]   |
| sequential  | concept uncertainty                    | 0.491 ± 0.021 | [0.439, 0.541]   |
| sequential  | 7-pt rule disagrees                    | 0.534 ± 0.013 | [0.505, 0.565]   |
| sequential  | hard vs soft concepts                  | 0.527 ± 0.026 | [0.489, 0.566]   |
| sequential  | LR: confidence only                    | 0.630 ± 0.017 | [0.583, 0.677]   |
| sequential  | LR: confidence + explanation           | 0.650 ± 0.012 | [0.605, 0.695]   |
| sequential  | Δ (explanation adds beyond confidence) |               | [-0.014, +0.058] |
| joint       | confidence (baseline)                  | 0.744 ± 0.014 | [0.712, 0.776]   |
| joint       | concept uncertainty                    | 0.690 ± 0.032 | [0.646, 0.729]   |
| joint       | 7-pt rule disagrees                    | 0.609 ± 0.036 | [0.581, 0.639]   |
| joint       | hard vs soft concepts                  | 0.682 ± 0.025 | [0.650, 0.712]   |
| joint       | LR: confidence only                    | 0.744 ± 0.014 | [0.712, 0.776]   |
| joint       | LR: confidence + explanation           | 0.758 ± 0.018 | [0.724, 0.792]   |
| joint       | Δ (explanation adds beyond confidence) |               | [-0.002, +0.031] |
