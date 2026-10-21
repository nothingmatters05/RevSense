# 🏁 RevSense --- Acoustic Engine Speed Estimation

> **Hear the rev. See the RPM.**\
> RevSense estimates motorcycle engine speed from recorded engine audio
> using digital signal processing (DSP).

RevSense is an educational DSP project that analyses motorcycle engine
sound, estimates its firing-event frequency, and converts that frequency
into an estimated engine speed in revolutions per minute (RPM). The
Streamlit dashboard lets you explore the recordings, play audio
alongside reference video where available, inspect waveforms and
spectra, and review the estimates produced across analysis windows.

------------------------------------------------------------------------

## 📌 Table of Contents

-   [✨ Project Overview](#-project-overview)
-   [🎯 Objectives](#-objectives)
-   [⚙️ How It Works](#️-how-it-works)
-   [🧮 Key Relationship](#-key-relationship)
-   [🖥️ Dashboard Features](#️-dashboard-features)
-   [📁 Project Structure](#-project-structure)
-   [🧰 Technologies Used](#-technologies-used)
-   [🚀 Run Locally](#-run-locally)
-   [☁️ Deploy to Streamlit Community
    Cloud](#️-deploy-to-streamlit-community-cloud)
-   [🎧 Adding Your Recordings](#-adding-your-recordings)
-   [📊 Understanding the Results](#-understanding-the-results)
-   [⚠️ Limitations](#️-limitations)
-   [🔭 Possible Future Improvements](#-possible-future-improvements)
-   [🤝 Contributing](#-contributing)
-   [📄 License](#-license)
-   [🙏 Acknowledgements](#-acknowledgements)

------------------------------------------------------------------------

## ✨ Project Overview

RevSense explores how engine sound can be used as a signal for
estimating engine speed. Instead of relying only on a physical RPM
sensor, the project analyses the frequency content of recorded audio.

The current demo is organised around recordings of a **Yamaha MT-15**
test motorcycle, including steady-speed recordings and a variable-speed
revving run. The dashboard presents the signal-processing process and
the results so that users can see how the estimate is formed.

**Project type:** Digital Signal Processing / Audio Signal Analysis\
**Interface:** Streamlit web application\
**Primary language:** Python

## 🎯 Objectives

-   🎙️ Analyse recorded engine audio using digital signal processing.
-   📈 Visualise the signal in the time domain and frequency domain.
-   🔎 Estimate the likely firing-event frequency using a harmonic-comb
    approach.
-   🧮 Convert the estimated firing frequency into RPM for a
    single-cylinder, four-stroke engine.
-   ⏱️ Analyse steady recordings in windows and track RPM through a
    revving recording.
-   🧪 Expose window-level estimates and quality indicators for
    inspection.
-   🌐 Present the analysis through a shareable interactive dashboard.

## ⚙️ How It Works

The estimator follows this high-level pipeline:

1.  **Load audio** --- read the WAV recording and convert stereo audio
    to mono when necessary.
2.  **Prepare the signal** --- remove the mean (DC offset) from the
    audio.
3.  **Split into windows** --- analyse short segments of the recording.
    The dashboard's current configuration uses longer windows for steady
    recordings and shorter, overlapping windows for the revving run.
4.  **Apply a Hann window** --- reduce spectral leakage near each
    segment's boundaries.
5.  **Compute the FFT** --- transform the audio segment into a frequency
    spectrum.
6.  **Evaluate harmonic candidates** --- score candidate firing
    frequencies by checking energy near their expected harmonics, rather
    than simply selecting the single largest spectral peak.
7.  **Convert frequency to RPM** --- use the engine model described
    below.
8.  **Check and summarise estimates** --- report window-level estimates,
    quality flags, and summary statistics.

### Steady recordings and revving recordings

-   **Steady recordings:** use overlapping analysis windows and
    summarise accepted estimates with statistics such as the median and
    percentile range.
-   **Revving recording:** uses shorter, more-overlapping windows to
    follow changes over time. The implementation also includes an
    octave-correction heuristic and flags some large adjacent-window
    changes for review.

These checks help identify suspicious estimates; they do not guarantee
that every estimate is correct.

## 🧮 Key Relationship

For a **single-cylinder, four-stroke engine**, one firing event occurs
per two crankshaft revolutions. If (f_0) is the firing-event frequency
in hertz (events per second), then:

\[ f_0 = `\frac{\mathrm{RPM}}{120}`{=tex} \]

Rearranging gives:

\[ `\boxed{\mathrm{RPM} = 120 f_0}`{=tex} \]

For example, a firing-event frequency of approximately
(33.3 `\text{Hz}`{=tex}) corresponds to:

\[ 120 `\times 33.3`{=tex} `\approx 3996`{=tex} `\mathrm{RPM}`{=tex} \]

This relationship depends on the assumed engine cycle and cylinder
configuration. The estimator's frequency estimate can still be affected
by noise, harmonics, and ambiguity in the recorded sound.

## 🖥️ Dashboard Features

The Streamlit dashboard includes:

-   🏍️ **Test-bike profile** --- background information about the
    motorcycle used for the recordings.
-   🎛️ **Recording selector** --- switch between the available
    steady-speed recordings and the revving run.
-   🔊 **Audio playback** --- play and stop the selected recording.
-   🎥 **Reference video** --- show the matching video when a supported
    video file is available; the reference video is muted.
-   🧭 **RPM dashboard** --- display the estimated RPM and, for steady
    recordings, the measured reference RPM.
-   🌊 **Waveform view** --- inspect audio amplitude over time.
-   📊 **Spectrum view** --- inspect the recording's frequency content.
-   🧪 **DSP workflow explanations** --- review the stages used by the
    estimator.
-   🔬 **Analysis-window inspector** --- inspect an individual window's
    waveform, spectrum, firing-frequency estimate, RPM, harmonic count,
    score, quality flag, and detected peaks.
-   📋 **Results and window-level outputs** --- review summary
    statistics and the per-window data used to form them.
-   📝 **Conclusion panel** --- view an automatically generated summary
    for the selected recording when audio is available.

## 📁 Project Structure

Keep the directory layout consistent with the paths used in the
application.

``` text
RevSense/
├── app.py                 # Streamlit dashboard and UI
├── rpm_estimator.py       # Audio analysis and RPM estimation
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
├── audio/
│   ├── 1500-1750.wav
│   ├── 2250.wav
│   ├── 3200.wav
│   ├── 4000.wav
│   ├── 5000.wav
│   ├── Revving.wav
│   └── matching video files (optional)
├── assets/
│   └── mt15.jpg            # Test motorcycle image (if included)
└── results/                # Reserved for project outputs, if used
```

The exact contents may differ depending on which media and output files
are included in your repository. Preserve the filenames expected by
`app.py`; the current code looks for the listed WAV names and matching
video files in the `audio/` directory.

## 🧰 Technologies Used

  -----------------------------------------------------------------------
  Technology                          Role in the project
  ----------------------------------- -----------------------------------
  Python                              Application and estimator
                                      implementation

  Streamlit                           Interactive web dashboard

  NumPy                               Numerical operations and
                                      FFT-related processing

  SciPy                               Signal-processing utilities,
                                      including Hann windows and peak
                                      detection

  Pandas                              Window-level data and summary
                                      tables

  SoundFile                           Reading audio recordings

  Matplotlib                          Waveform and spectrum plots

  HTML, CSS and JavaScript            Embedded media player and animated
                                      tachometer interface
  -----------------------------------------------------------------------

## 🚀 Run Locally

### 1. Prerequisites

Install a supported version of Python and Git. Make sure the project
files and required media folders are available locally.

### 2. Clone the repository

Replace `YOUR-GITHUB-USERNAME` with your GitHub username and
`YOUR-REPOSITORY` with the repository name:

``` bash
git clone https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

If you already have the project on your computer, open a terminal in
that project directory instead.

### 3. Create and activate a virtual environment (recommended)

**Windows --- Command Prompt:**

``` bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project's `requirements.txt` should include:

``` text
streamlit>=1.36
numpy
pandas
soundfile
matplotlib
scipy
```

### 5. Start the dashboard

``` bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open
it in your browser.

## ☁️ Deploy to Streamlit Community Cloud

RevSense can be deployed from a GitHub repository using Streamlit
Community Cloud.

1.  Push the complete project to GitHub.
2.  Confirm that `app.py`, `rpm_estimator.py`, and `requirements.txt`
    are at the repository root.
3.  Confirm that the audio files and required assets have been uploaded
    to the expected directories.
4.  Visit [Streamlit Community Cloud](https://share.streamlit.io/) and
    sign in with GitHub.
5.  Create an app and select the repository, branch, and `app.py` as the
    main file.
6.  Deploy the app and wait for the dependency installation and startup
    process to finish.
7.  Open the deployed URL and test the recordings, plots, dashboard, and
    analysis results.

### Deployment notes

-   The hosted app can only access files that are included in the
    repository or otherwise made available to it.
-   Large audio and video files can increase repository size, deployment
    time, and memory use.
-   GitHub's browser uploader has file-size limits. Check GitHub's
    current documentation if media files are too large; Git LFS or a
    suitable external media-storage strategy may be needed.
-   If deployment fails, inspect the Streamlit application logs and
    check for missing files, dependency issues, or incorrect paths.
-   Do not commit credentials, access tokens, private keys, or other
    secrets to the repository.

Useful links: - [Streamlit Community
Cloud](https://share.streamlit.io/) - [Streamlit deployment
documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) -
[GitHub documentation](https://docs.github.com/)

## 🎧 Adding Your Recordings

1.  Put the expected WAV recordings in the `audio/` directory.
2.  Keep the filenames consistent with those referenced by `app.py`.
3.  To display a matching video, place it in the same `audio/` directory
    and use the same filename stem with a supported extension (`.mp4`,
    `.mov`, `.webm`, or `.m4v`).
4.  Place the motorcycle image at `assets/mt15.jpg` if you want the
    image shown by the dashboard.
5.  Restart the app or refresh it after updating files.

The application creates the `audio/`, `results/`, and `assets/`
directories if they do not exist, but creating those directories does
not supply the actual recording or image files.

## 📊 Understanding the Results

Depending on the selected recording and available data, the dashboard
can show:

-   **Estimated RPM:** the engine speed inferred from the estimated
    firing-event frequency.
-   **Reference / actual RPM:** the supplied measured reference for the
    steady-speed recordings. This is a comparison value, not a value
    measured by the audio estimator.
-   **Median RPM:** the median of the estimates included in the relevant
    summary.
-   **P10 and P90:** the 10th and 90th percentiles, describing the
    spread of estimates.
-   **Accepted windows:** the number of windows counted as accepted by
    the estimator's quality/continuity logic.
-   **Matched harmonics and harmonic score:** indicators of how much
    spectral evidence supports a candidate frequency.
-   **Quality flag:** a diagnostic label indicating whether the
    implementation's checks passed or whether the window needs review.
-   **Window-level outputs:** the individual estimates and diagnostic
    fields for each analysis window.

Interpret results together with the quality flags and the recording
conditions. A narrow percentile range does not by itself prove that an
estimate matches the true engine speed.

## ⚠️ Limitations

RevSense is an educational signal-processing project, not a certified
automotive measurement instrument.

-   Audio-based RPM estimates can be affected by background noise,
    microphone placement, recording level, exhaust and mechanical
    resonances, and other sound sources.
-   The strongest spectral peak may be a harmonic rather than the
    fundamental firing frequency; the harmonic-comb approach is intended
    to reduce this risk but cannot eliminate it.
-   Octave ambiguity can lead to an estimate near twice or half the
    intended frequency.
-   Short windows respond faster to changing RPM but provide less
    frequency resolution than longer windows.
-   Quality flags are heuristic diagnostics, not a guarantee of
    correctness.
-   The accuracy of a result should be assessed against a trusted
    reference RPM measurement under the same recording conditions.
-   Results depend on the quality and duration of the input audio and on
    the implementation's configured RPM range and analysis parameters.

## 🔭 Possible Future Improvements

Potential future work could include:

-   Validate estimates against a trusted tachometer across more speeds
    and recording conditions.
-   Add confidence metrics calibrated against measured errors.
-   Improve octave-error detection and temporal tracking for rapid
    acceleration.
-   Evaluate microphone distance, placement, background noise, and
    recording levels systematically.
-   Add automated test cases and regression tests for the estimator.
-   Add clearer export options for research results and experiment
    reports.
-   Evaluate performance and media-loading behaviour on the hosted
    deployment.

These are possible extensions, not claims about features already
implemented.

## 🤝 Contributing

Suggestions, bug reports, and improvements are welcome.

1.  Fork the repository.
2.  Create a branch for your change.
3.  Make and test your changes.
4.  Open a pull request describing the change and its purpose.

For changes to the RPM estimator, include the recording conditions and
reference measurements used to validate the change whenever possible.

## 📄 License

No license has been specified here. Until a license is added to the
repository, reuse and redistribution permissions are not explicitly
granted. If you plan to publish the project for others to reuse, choose
an appropriate license and add a `LICENSE` file.

## 🙏 Acknowledgements

Built as a learning project exploring digital signal processing, audio
analysis, frequency-domain methods, and interactive engineering
visualisation.

**RevSense --- Hear the rev. See the RPM.** 🏁
