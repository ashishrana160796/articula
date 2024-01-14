#!/bin/bash

# Function to run script with specific arguments
function run_script {
    echo "Running script with arguments: $@"
    python ensemble_prediction_pipeline.py "$@"
    echo "Finished running script with arguments."
    echo "------------------------"
}

# Run with individual arguments for Gaussian noise
run_script --gaussian_noise --gaus_intensity 5
run_script --gaussian_noise --gaus_intensity 10
run_script --gaussian_noise --gaus_intensity 20

# Run with individual arguments for Frequency
run_script --freq --freq_prob 0.1
run_script --freq --freq_prob 0.2
run_script --freq --freq_prob 0.5

# Run with individual arguments for Salt & Pepper Noise
run_script --salt_pepper_noise --s_p_density 0.1
run_script --salt_pepper_noise --s_p_density 0.2
run_script --salt_pepper_noise --s_p_density 0.3

# Run with combinations of arguments
run_script --freq --freq_prob 0.1 --gaussian_noise --gaus_intensity 5 --salt_pepper_noise --s_p_density 0.1
run_script --freq --freq_prob 0.2 --gaussian_noise --gaus_intensity 10 --salt_pepper_noise --s_p_density 0.2
run_script --freq --freq_prob 0.5 --gaussian_noise --gaus_intensity 20 --salt_pepper_noise --s_p_density 0.3
run_script --gaussian_noise --gaus_intensity 10 --grains_texture --grains_intensity 10
run_script --freq --freq_prob 0.1 --salt_pepper_noise --s_p_density 0.3

# Add more combinations as needed

echo "All combinations completed."
