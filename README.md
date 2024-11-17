# Performance of OFDM in Energy Detection and Cyclostationary Detection for Cognitive Radio

This project simulates the performance of Orthogonal Frequency Division Multiplexing (OFDM) in Energy Detection for Cognitive Radio (CR) systems. The goal is to assess the detection capabilities of a cognitive radio system, particularly using energy detection, while generating a dataset with both primary user (PU) activity and detection results. The dataset can be used to study the relationship between Signal-to-Noise Ratio (SNR), False Alarm Probability (P_FA), and Detection Probability (P_D).

# Table of Contents

Overview

Technologies Used


Code Structure


Results and Visualization


# Overview

This project uses the OFDM modulation technique to simulate communication signals and employs Energy Detection to detect the presence of primary users (PU) in a Cognitive Radio system. The program generates a dataset containing real and imaginary components of the received signal, PU activity, and detection results.
The dataset is used to evaluate the performance of energy detection across different SNR values and calculate performance metrics like False Alarm Probability (P_FA) and Detection Probability (P_D).

# Technologies Used

Python 3.x: The primary programming language used for simulation and signal processing.

NumPy: Used for numerical operations and signal processing.

Pandas: For creating and manipulating datasets.

Matplotlib: For plotting histograms and other visualizations.

SciPy: For signal processing functions like Inverse Fast Fourier Transform (IFFT).

Scikit-learn (optional): Can be used for further statistical analysis (if needed).

# Code Structure

qpsk_mod(bits): Performs QPSK modulation of binary bits.

generate_ofdm_symbol(bits): Generates an OFDM symbol with the QPSK modulated bits.

generate_ofdm_message(num_symbols): Generates a sequence of OFDM symbols (a message) by generating multiple symbols.

add_noise(ofdm_signal, SNR_dB): Adds Gaussian noise to the OFDM signal based on the given SNR in dB.

energy_detection(received_signal, threshold): Performs energy detection by comparing the energy of the received signal with a threshold.

create_dataset(num_samples, snr_list, output_file): Generates a dataset of OFDM signals, including PU activity and detection results.

plot_dataset(csv_file, detection_results): Reads the generated CSV dataset, plots histograms of PU activity and detection results, and plots the detection performance (P_FA and P_D) vs. SNR.

# Results and Visualization
The program generates the following visualizations:

Histogram of PU Activity: Shows how often the primary user is active (1) or inactive (0).

Histogram of Detection Results: Shows how often the energy detection method successfully detects the presence of a primary user.

Scatter Plot of Real vs Imaginary Parts of OFDM Signal: Visualizes the OFDM signal's complex values in a scatter plot.

Detection Performance (P_Fa and P_D vs. SNR): Plots the False Alarm Probability (P_Fa) and Detection Probability (P_D) as a function of SNR.

[Capture](https://github.com/user-attachments/assets/75457838-a5cb-41de-a5b1-12b88ca51d4b)

[Capture4](https://github.com/user-attachments/assets/d693bd7c-4e37-4ef4-98ea-a0115e032d8a)

[Capture3](https://github.com/user-attachments/assets/624ed2cf-f094-4043-a33c-14f8c9390473)

[Capture7](https://github.com/user-attachments/assets/9e380527-2590-4826-82f7-0b9b2ad61a3e)














