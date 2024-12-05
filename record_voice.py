import sounddevice as sd
from scipy.io.wavfile import write
import os
import datetime

# Folder to save recorded audio files
output_folder = "recorded_audio"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

def record_voice(duration=5, sample_rate=44100):
    print("Recording... Speak now!")
    # Capture audio from the microphone
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    
    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"recording_{timestamp}.wav"
    file_path = os.path.join(output_folder, file_name)
    
    # Save the recorded audio
    write(file_path, sample_rate, audio_data)
    print(f"Recording saved as {file_path}")
    
    return file_path

# Example usage
if __name__ == "__main__":
    duration = int(input("Enter recording duration (in seconds): "))
    recorded_file = record_voice(duration=duration)
    print(f"Recorded file saved at: {recorded_file}")
