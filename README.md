# RevSense — Sporty red/black GUI

## Run
1. Keep `app.py`, `rpm_estimator.py`, and `requirements.txt` together.
2. Put the six WAV files in `audio/`.
3. Optional corresponding videos should use the same basename (e.g. `5000.mp4` for `5000.wav`).
4. In VS Code terminal run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Included
- Sporty red/black theme and automotive semicircular tachometer with needle, numbered ticks and redline.
- Audio playback, optional matching video, waveform and frequency spectrum.
- Window-specific DSP walkthrough with outputs and short theory explanations.
- Harmonic evidence, f0-to-RPM conversion, quality table, median/P10/P90 definitions and conclusion.
- On-page result tables; CSV downloads are optional.

## Notes
The PLAY ESTIMATE button starts audio and renders an illustrative gauge ramp to the recording median. This is not live RPM tracking, and browser audio/video/gauge playback is not frame-synchronized. The reference gauge shows the approximate dataset label, not independently measured ground truth. `Revving.wav` supports media and signal views, but frame-by-frame variable-RPM estimation is not enabled yet.
