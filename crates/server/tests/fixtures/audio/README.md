# Audio fixtures

One-second 440 Hz sine tones, generated once with ffmpeg (no recording of a
person, nothing from `H:\speech-study`). They feed `speech::audio`'s decode
tests and the transcription route's format tests.

```
S='sine=frequency=440:duration=1:sample_rate=44100'
ffmpeg -f lavfi -i "$S" -ac 1 -ar 16000 -c:a pcm_s16le tone.wav
ffmpeg -f lavfi -i "$S" -ac 2 -ar 44100 -c:a pcm_s16le tone-44k-stereo.wav
ffmpeg -f lavfi -i "$S" -ac 1 -ar 48000 -c:a flac tone.flac
ffmpeg -f lavfi -i "$S" -ac 2 -c:a libmp3lame -b:a 128k tone.mp3
ffmpeg -f lavfi -i "$S" -ac 1 -c:a aac -b:a 64k -f adts tone.aac
ffmpeg -f lavfi -i "$S" -ac 1 -c:a aac -b:a 64k tone.m4a
ffmpeg -f lavfi -i "$S" -ac 1 -ar 16000 -c:a flac -f ogg tone-flac.ogg
ffmpeg -f lavfi -i "$S" -ac 1 -ar 16000 -c:a flac -f matroska tone.mka
ffmpeg -f lavfi -i "$S" -ac 1 -ar 48000 -c:a libopus -b:a 32k -f ogg tone-opus.ogg
ffmpeg -f lavfi -i "$S" -ac 1 -ar 48000 -c:a libopus -b:a 32k -f webm tone-opus.webm
```
