import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import time
import pygame
from tkinter import filedialog, messagebox

# Assuming kalman_filter.py is in the same directory
from kalman_filter import kalman_filter  

# Initialize pygame mixer
pygame.mixer.init()

# Global variables to store voice data and sample rate
voice_data = None
sample_rate = None
loaded_voice = None
filtered_filename = None
noisy_voice_data = None  # Variable to store noisy voice data

# Create a global variable to keep track of the current playing voice
current_voice = None
is_voice_playing = False  # Flag to track if a voice is currently playing

def add_voice():
    global voice_data, sample_rate, loaded_voice, filtered_filename, is_voice_playing
    # Load voice data
    filename = filedialog.askopenfilename(title="Select WAV file")
    if filename:
        sample_rate, voice_data = wavfile.read(filename)
        loaded_voice = filename

        # Enable play voice button
        play_stop_button.config(state="normal")
        # Enable add noise button
        add_noise_button.config(state="normal")
        # Disable play filtered button
        play_filtered_button.config(state="disabled")
        # Enable play noisy button
        play_noise_button.config(state="disabled")

def add_gaussian_noise(audio, std_dev):
    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, len(audio))
    # Add noise to the audio
    noisy_audio = audio + noise
    return noisy_audio

def calculate_snr(audio, noisy_audio):
    # Calculate signal power
    signal_power = np.sum(audio ** 2)
    # Calculate noise power
    noise_power = np.sum((noisy_audio - audio) ** 2)
    # Check if noise power is zero or negative
    if noise_power <= 0:
        return "SNR cannot be calculated: Invalid noise power"
    else:
        # Calculate SNR
        SNP = np.abs(signal_power / noise_power)
        snr = 10 * np.log10(SNP)
        return snr

def play_voice():
    global current_voice, loaded_voice, is_voice_playing
    if loaded_voice and not is_voice_playing:
        is_voice_playing = True
        current_voice = pygame.mixer.Sound(loaded_voice)
        current_voice.play()

def stop_voice():
    global current_voice, is_voice_playing
    if current_voice:
        pygame.mixer.stop()
        is_voice_playing = False

def start_filter():
    global voice_data, sample_rate, filtered_filename
    if voice_data is not None:
        Q = 1e-3
        R = 0.1

        # Start time for Kalman filtering
        start_time = time.time()

        # Apply Kalman filter
        filtered_voice = kalman_filter(voice_data, Q, R)

        # End time for Kalman filtering
        end_time = time.time()
        filter_time = end_time - start_time

        # Save filtered voice data with a ".wav" extension
        filtered_filename = filedialog.asksaveasfilename(title="Save Filtered Voice As", filetypes=[("WAV files", "*.wav")], defaultextension=".wav")
        if filtered_filename:
            wavfile.write(filtered_filename, sample_rate, filtered_voice.astype(np.int16))

            # Enable play filtered button
            play_filtered_button.config(state="normal")

            # Plot original and filtered voice signals
            plt.figure(figsize=(7, 7))
            plt.plot(np.arange(len(voice_data)) / sample_rate, voice_data, label='Original Voice Signal', color='b', alpha=0.5)
            plt.plot(np.arange(len(filtered_voice)) / sample_rate, filtered_voice, label='Filtered Voice Signal', color='r')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Original vs Filtered Voice Signal')
            plt.legend()
            plt.grid(True)

            # Calculate SNR for the filtered signal
            snr_filtered = calculate_snr(voice_data, filtered_voice)
            plt.text(0.05, 0.85, f'SNR: {snr_filtered:.2f} dB', transform=plt.gca().transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            # Display filtering time
            plt.text(0.05, 0.80, f'Filtering Time: {filter_time:.2f} seconds', transform=plt.gca().transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            plt.show()

def add_noise_and_plot():
    global voice_data, sample_rate, noisy_voice_data
    if voice_data is not None:
        # Define parameters for Gaussian noise
        std_dev = 0.1  # You can adjust this value as needed

        # Add Gaussian noise to the voice data
        noisy_voice_data = add_gaussian_noise(voice_data, std_dev)

        # Plot original, noisy, and noisy voice signals
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(voice_data)) / sample_rate, voice_data, label='Original Voice Signal', color='b', alpha=0.7)
        plt.plot(np.arange(len(noisy_voice_data)) / sample_rate, noisy_voice_data, label='Noisy Voice Signal', color='r', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Original vs Noisy Voice Signals')
        plt.legend()
        plt.grid(True)

        # Calculate SNR for the noisy signal
        snr_noisy = calculate_snr(voice_data, noisy_voice_data)
        plt.text(0.05, 0.85, f'SNR: {snr_noisy:.2f} dB', transform=plt.gca().transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # Enable play noise button
        play_noise_button.config(state="normal")

        plt.show()
    else:
        tk.messagebox.showerror("Error", "Please add a voice file first.")

def plot_snr_vs_time_with_noise():
    global voice_data
    if voice_data is not None:
        std_dev = 0.1  # Standard deviation of the Gaussian noise
        snr_values = []
        time_values = []

        # Adding noise and calculating SNR over time
        for t in range(1, 11):  # Simulate over 10 seconds
            noisy_voice_data = add_gaussian_noise(voice_data, std_dev * t)
            snr_noisy = calculate_snr(voice_data, noisy_voice_data)
            snr_values.append(snr_noisy)
            time_values.append(t)

        # Plot SNR vs Time with Noise
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, snr_values, 'bo-')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs Time with Noise')
        plt.grid(True)
        plt.show()
    else:
        tk.messagebox.showerror("Error", "Please add a voice file first.")

def play_stop_filtered_voice():
    global is_voice_playing, current_voice, filtered_filename
    if pygame.mixer.get_busy() and current_voice:
        stop_voice()
        play_filtered_button.config(text="Play Filtered Voice")
    else:
        if filtered_filename:
            is_voice_playing = True
            current_voice = pygame.mixer.Sound(filtered_filename)
            current_voice.play()
            play_filtered_button.config(text="Stop Filtered Voice")

def play_stop_noisy_voice():
    global is_voice_playing, current_voice, noisy_voice_data
    if is_voice_playing:
        stop_voice()
        is_voice_playing = False
        play_noise_button.config(text="Play Noisy Voice")
    else:
        if noisy_voice_data is not None:
            is_voice_playing = True
            current_voice = pygame.mixer.Sound(noisy_voice_data.tobytes())
            current_voice.play()
            play_noise_button.config(text="Stop Noisy Voice")

root = tk.Tk()
root.title("Voice Filter")
root.geometry("500x750")
root.resizable(False, False)

#dffddfdsfsf////////////////////////////////////////////////////////////////////////////////////////////
image = Image.open("./image/bgdd.jpg")
image = image.resize((500, 710), Image.ANTIALIAS if hasattr(Image, "ANTIALIAS") else Image.BILINEAR)
bg_image = ImageTk.PhotoImage(image)

bg_label = tk.Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

root.configure(bg="white")

label = tk.Label(root, text="kalman_filter", font=("Arial", 18), fg="black")

label.pack(pady=10)

add_button = tk.Button(root, text="Add Voice", font=12, command=add_voice, bg="blue", fg="white", height=2, width=40, anchor="center", borderwidth=12)
add_button.pack(pady=10)

play_stop_button = tk.Button(root, text="Play/Stop Voice", font=12, command=lambda: [play_voice() if not pygame.mixer.get_busy() and not is_voice_playing else stop_voice()], bg="blue", fg="white", height=2, width=40, anchor="center", borderwidth=12)
play_stop_button.pack(pady=10)
play_stop_button.config(state="disabled")  # Initially disabled

add_noise_button = tk.Button(root, text="Add Gaussian Noise and Plot", font=12, command=add_noise_and_plot, bg="blue", fg="white", height=2, width=40, anchor="center", borderwidth=12)
add_noise_button.pack(pady=10)

play_noise_button = tk.Button(root, text="Play/Stop Noisy Voice", font=12, command=play_stop_noisy_voice, bg="blue", height=2, width=40, borderwidth=12)
play_noise_button.pack(pady=10)
play_noise_button.config(state="disabled")  # Initially disabled

start_button = tk.Button(root, text="Start Kalman Filter", font=12, command=start_filter, height=2, width=40, bg="blue", fg="white", borderwidth=12)
start_button.pack(pady=10)

play_filtered_button = tk.Button(root, text="Play Filtered Voice", font=12, command=play_stop_filtered_voice, bg="blue", height=2, width=40, borderwidth=12)
play_filtered_button.pack(pady=10)
play_filtered_button.config(state="disabled")  # Initially disabled

plot_snr_vs_time_with_noise_button = tk.Button(root, text="Plot SNR vs Time with Noise", font=12, command=plot_snr_vs_time_with_noise, height=2, width=40, bg="blue", fg="white", borderwidth=12)
plot_snr_vs_time_with_noise_button.pack(pady=10)

root.mainloop()

