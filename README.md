# WHAMSAH
## We Have [Redacted] Sing At Home!
### (where redacted might stay for Apple Music)
WHAMSAH is an audio application capable of performing voice removal/isolation with a low-latency (about 1.5 seconds).
The separation is powered by a custom U-Net architecture (aka WHAMSAHNet) with a bottleneck layer inspired by [Apple's singing voice cancellation model](https://arxiv.org/abs/2401.12068); WHAMSAHNet was trained and tested solely on MUSDB18-HQ - no extra data was used.

The current model reached an SDR of 4.71 dB on the MUSDB18 test set.

## Known issues
* Noisy separation
* Faint clicks when passing from one frame to the next
* Separation performance degrades with fully stereo content

## Requirements
```
einops
librosa
numpy
pillow
sounddevice
soundfile
torch
```
