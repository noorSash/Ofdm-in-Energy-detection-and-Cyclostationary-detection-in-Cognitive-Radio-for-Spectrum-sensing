import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.stats import chi2
# مهم 2
# Parameters
N = 64
CP_len = 16
K = 10
SNR_list = np.arange(0, 21, 5)
sigma_s = 1
P_FA = 0.1
output_csv_file = 'ofdm_pu_dataset.csv'

def qpsk_mod(bits):
    return (1/np.sqrt(2)) * ((2*bits[:, 0] - 1) + 1j*(2*bits[:, 1] - 1))

def generate_ofdm_symbol(bits):
    symbols = qpsk_mod(bits)
    ofdm_time = ifft(symbols, n=N)
    cp = ofdm_time[-CP_len:]
    ofdm_time_cp = np.concatenate([cp, ofdm_time])
    return ofdm_time_cp

def generate_ofdm_message(num_symbols):
    message = []
    for _ in range(num_symbols):
        bits = np.random.randint(0, 2, (N, 2))
        symbol = generate_ofdm_symbol(bits)
        message.append(symbol)
    return np.concatenate(message)

def add_noise(ofdm_signal, SNR_dB):
    SNR_linear = 10**(SNR_dB / 10)
    signal_power = np.mean(np.abs(ofdm_signal)**2)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_signal.shape) + 1j * np.random.randn(*ofdm_signal.shape))
    return ofdm_signal + noise

def energy_detection(received_signal, threshold):
    energy = np.sum(np.abs(received_signal)**2)
    return energy > threshold

def create_dataset(num_samples, snr_list, output_file):
    data = []
    detection_results = []
    for _ in range(num_samples):
        ofdm_message = generate_ofdm_message(K)
        SNR = np.random.choice(snr_list)
        ofdm_message_noisy = add_noise(ofdm_message, SNR)
        pu_activity = np.random.choice([0, 1])
        if pu_activity == 1:
            pu_signal = np.random.randn(len(ofdm_message)) + 1j * np.random.randn(len(ofdm_message))
            pu_signal_noisy = add_noise(pu_signal, SNR)
            combined_signal = ofdm_message_noisy + pu_signal_noisy
        else:
            combined_signal = ofdm_message_noisy
        threshold = chi2.ppf(1 - P_FA, df=N)
        detection_result = energy_detection(combined_signal, threshold)
        data.append(np.concatenate([np.real(combined_signal), np.imag(combined_signal), [pu_activity, detection_result]]))
        detection_results.append((SNR, pu_activity, detection_result))
    
    # Creating header names
    num_signal_columns = len(np.real(combined_signal))
    real_headers = [f'Real_{i}' for i in range(num_signal_columns)]
    imag_headers = [f'Imag_{i}' for i in range(num_signal_columns)]
    headers = real_headers + imag_headers + ['PU_Activity', 'Detection_Result']

    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_file, index=False)
    return detection_results

def plot_dataset(csv_file, detection_results):
    df = pd.read_csv(csv_file, header=0)  # Now including the header
    combined_signals_real = df.iloc[:, :-2].values
    pu_activity = df.iloc[:, -2].values
    detection_result = df.iloc[:, -1].values
    
    # Plot histogram of PU activity
    plt.figure()
    plt.hist(pu_activity, bins=2, edgecolor='black')
    plt.title('Histogram of PU Activity')
    plt.xlabel('PU Activity')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Inactive', 'Active'])
    plt.savefig('pu_activity_histogram.png')
    plt.show()
    
    # Plot histogram of energy detection results
    plt.figure()
    plt.hist(detection_result, bins=2, edgecolor='black')
    plt.title('Histogram of Energy Detection Results')
    plt.xlabel('Detection Result')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Detection', 'Detection'])
    plt.savefig('detection_result_histogram.png')
    plt.show()
    
    # Scatter plot of the real vs imaginary parts of OFDM signals
    plt.figure()
    plt.scatter(combined_signals_real[:, 0], combined_signals_real[:, 1], alpha=0.5, s=1)
    plt.title('Scatter Plot of OFDM Signal (Real vs Imaginary)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.savefig('ofdm_signal_scatter.png')
    plt.show()

    # Calculate and plot P_Fa and P_D
    snr_list = np.unique([d[0] for d in detection_results])
    P_Fa_list = []
    P_D_list = []

    for SNR in snr_list:
        detections = np.array([d[2] for d in detection_results if d[0] == SNR])
        pu_activities = np.array([d[1] for d in detection_results if d[0] == SNR])

        # Calculate False Alarm Probability
        P_Fa = np.mean((pu_activities == 0) & (detections == 1))
        # Calculate Detection Probability
        P_D = np.mean((pu_activities == 1) & (detections == 1))

        print(f'SNR: {SNR}, P_Fa: {P_Fa}, P_D: {P_D}')  # Print the values for each SNR

        P_Fa_list.append(P_Fa)
        P_D_list.append(P_D)

    plt.figure()
    plt.plot(snr_list, P_Fa_list, marker='o', label='P_Fa (False Alarm Probability)')
    plt.plot(snr_list, P_D_list, marker='o', label='P_D (Detection Probability)')
    plt.title('P_Fa and P_D vs. SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('pf_pd_vs_snr.png')
    plt.show()

# Generate dataset and plot
detection_results = create_dataset(num_samples=1000, snr_list=SNR_list, output_file=output_csv_file)
plot_dataset(output_csv_file, detection_results)




