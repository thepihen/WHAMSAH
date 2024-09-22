# WHAMSAH
## We Have [Redacted] Sing At Home!
### (where redacted might stay for Apple Music)
WHAMSAH is an audio application capable of performing voice removal/isolation from either a sound file or an audio stream with a low-latency (about 1.5 seconds) and a small memory footprint.

The script currently requires about 1.5 GB of VRAM and 8GB of RAM. It was verified to be working with no issues on an NVIDIA 1650 Super.

## Usage
WHAMSAH has two working modes, sync and async. Both separate audio in real-time, the difference between the two is that the former uses an audio stream to get new input data for the model, the latter instead pre-loads an audio file and reads chunks from it.

Sync mode requires the user to first select a Host API (e.g. MME, DirectSound, WASAPI), then an input and an output device - it is up to the user to ensure the correct API is selected and the devices are properly configured (streaming in stereo rather than mono, proper sample rate, etc.). Of course an audio source must be writing audio data to the input device in order for it to be picked up by WHAMSAH; while ensuring this is the case for an external device (e.g. a microphone) is easy, if the source is an application this can be setup e.g. on the Windows sound settings by choosing the WHAMSAH input device as the output for the other source.

Async mode requires the user to first select a source using the file picker. The app currently supports FLAC, MP3 and WAV audio files. In this case, selecting an API and input/output device combination is not required as the script will always use the system defaults.

For both modes, two sliders are available to control the loudness of both the vocals and the instrumental track.


## Requirements
Sync mode requires a virtual audio cable in order to separate audio from another application. A virtual audio cable is not already included in this repository, it is up to the user to download and install one.

The following Python libraries are required:
```
einops
librosa
numpy
pillow
sounddevice
soundfile
torch
```

## Model details
The separation is powered by a custom U-Net architecture (aka WHAMSAHNet) with a bottleneck layer inspired by [Apple's singing voice cancellation model](https://arxiv.org/abs/2401.12068); WHAMSAHNet was trained and tested solely on MUSDB18-HQ - no extra data was used.

The current model reached an SDR of 5.21 dB on the MUSDB18 test set.

## Known issues with the current model
* Noisy separation
* Faint clicks when passing from one frame to the next
* Separation performance degrades with fully stereo content
