\# Deep Learning V3 — Reproducibility Run Guide



\## 1. Purpose



This guide provides the steps required to reproduce the Deep Learning V3 EEG seizure-detection experiments contained in this repository.



The repository contains the source code, configuration, model definitions, training code, preprocessing scripts, and analysis scripts used for the experiments.



The CHB-MIT Scalp EEG dataset and generated experiment data are not included in this repository.



\---



\# 2. Experimental Cohort



The experiments use a 21-patient cohort:



```text

chb01

chb02

chb03

chb04

chb05

chb06

chb07

chb08

chb09

chb10

chb11

chb12

chb13

chb14

chb15

chb16

chb17

chb18

chb19

chb20

chb22

```



`chb21` is excluded because it is a re-recording of `chb01`.



`chb23` and `chb24` are also outside the experimental cohort.



The evaluation uses leave-one-patient-out cross-validation (LOOCV).



\---



\# 3. Main Signal Configuration



The main Deep Learning V3 pipeline uses:



\* Sampling frequency: 256 Hz

\* Window length: 4 seconds

\* Window size: 1024 samples

\* Window overlap: 50%

\* Window stride: 512 samples

\* Main bandpass: 0.5–50 Hz

\* Notch frequency: 60 Hz

\* Number of EEG channels: 18

\* Channel representation: common bipolar montage



The 18 bipolar channels are:



```text

FP1-F7

F7-T7

T7-P7

P7-O1



FP1-F3

F3-C3

C3-P3

P3-O1



FP2-F4

F4-C4

C4-P4

P4-O2



FP2-F8

F8-T8

T8-P8

P8-O2



FZ-CZ

CZ-PZ

```



The random seed used by the pipeline is:



```text

42

```



\---



\# 4. Actual Experimental Environment



The following environment corresponds to the environment used for the Deep Learning V3 experiments.



\## Environment location



```text

D:\\seizure\_dl\_v3\\seizure\_dl\_v3\\.venv

```



\## Software versions



```text

Python:        3.13.3

PyTorch:       2.6.0+cu124

CUDA:          12.4

NumPy:         2.4.4

SciPy:         1.17.1

Pandas:        3.0.2

scikit-learn:  1.8.0

MNE:           1.12.1

```



\## Hardware



```text

GPU:           NVIDIA GeForce RTX 4070 Laptop GPU

CUDA ready:    True

```



These versions document the actual computational environment used for the experiments. They are provided for reproducibility and may differ from the minimum or range specifications in `requirements.txt`.



\---



\# 5. Repository Structure



The main repository structure is:



```text

seizure\_dl\_v3/

│

├── README.md

├── RUN\_GUIDE.md

├── requirements.txt

│

├── scripts/

│   ├── README\_STAGES.md

│   ├── analyze\_dl\_results.py

│   ├── analyze\_stages.py

│   ├── diagnostic\_v3.py

│   ├── prepare\_raw\_segments.py

│   ├── prepare\_raw\_segments\_25hz.py

│   ├── run\_classical\_perseg.py

│   ├── run\_dl\_loocv.py

│   ├── run\_dl\_loocv\_25hz.py

│   ├── run\_dl\_loocv\_perrec.py

│   ├── run\_smoke\_test.py

│   └── verify\_gpu.py

│

└── src/

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── config.py

&#x20;   ├── models.py

&#x20;   └── training.py

```



\---



\# 6. Python Environment Setup



Open PowerShell in the repository root.



Create a virtual environment:



```powershell

python -m venv .venv

```



Activate the environment:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Confirm that Python is available:



```powershell

python --version

```



The experimental environment used Python 3.13.3.



\---



\# 7. Install PyTorch



The actual experimental environment used:



```text

PyTorch 2.6.0+cu124

CUDA 12.4

```



For a new installation, install a CUDA-enabled PyTorch build compatible with the local NVIDIA driver and GPU.



The exact PyTorch installation command may depend on the CUDA wheel currently provided by PyTorch.



For example, the `requirements.txt` comments specify a CUDA 12.1 installation command:



```powershell

python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

```



This command is an installation example from the project requirements. It is not a claim that the original experimental environment used CUDA 12.1.



The original experiments were run with:



```text

PyTorch 2.6.0+cu124

CUDA 12.4

```



\---



\# 8. Install Python Dependencies



After installing PyTorch, install the dependencies listed in `requirements.txt`:



```powershell

python -m pip install -r requirements.txt

```



The repository's `requirements.txt` specifies the required scientific, EEG-processing, visualization, and deep-learning packages.



For exact reproduction of the reported experiments, the actual environment versions documented in Section 4 should be used as the reference environment.



\---



\# 9. Verify GPU Availability



Run:



```powershell

python scripts\\verify\_gpu.py

```



The expected experimental environment reported:



```text

GPU: NVIDIA GeForce RTX 4070 Laptop GPU

CUDA ready: True

```



A CUDA-capable NVIDIA GPU is recommended for the deep-learning experiments.



Stage 3 is CPU-bound and does not require a GPU.



\---



\# 10. Obtain the CHB-MIT Dataset



The CHB-MIT Scalp EEG dataset is not included in this repository.



The dataset is publicly available through PhysioNet:



https://physionet.org/content/chbmit/1.0.0/



Download the dataset separately and store it locally.



Do not upload the CHB-MIT dataset to GitHub.



\---



\# 11. Configure the Dataset Location



The dataset location is defined in:



```text

src\\config.py

```



The configuration contains:



```python

DATA\_ROOT = Path(...)

```



Set `DATA\_ROOT` to the local directory containing the CHB-MIT dataset.



For example:



```python

DATA\_ROOT = Path(r"D:\\data\\chbmit")

```



Use the actual dataset location on the computer performing the reproduction.



The source code and experiment configuration should otherwise be kept unchanged when reproducing the supplied experiments.



\---



\# 12. Stage 1 — EEGNet with Per-Segment Z-Score



\## Purpose



Stage 1 is the main Deep Learning V3 EEGNet experiment.



The training pipeline applies per-segment, per-channel z-score normalization.



The experiment uses 21-patient LOOCV.



\---



\## 12.1 Prepare the Main Raw Cache



Run:



```powershell

python scripts\\prepare\_raw\_segments.py

```



This preprocessing script:



\* reads the CHB-MIT EDF recordings;

\* selects the common bipolar channels;

\* applies the main signal preprocessing;

\* uses a 0.5–50 Hz bandpass;

\* applies a 60 Hz notch filter;

\* uses 256 Hz sampling;

\* creates 4-second windows;

\* uses 50% overlap;

\* identifies seizure and baseline segments;

\* creates the raw segment cache used by Stage 1.



The generated cache consists of large `.npz` files and should remain outside the Git repository.



\---



\## 12.2 Run Stage 1 EEGNet



Run the complete EEGNet LOOCV experiment:



```powershell

python scripts\\run\_dl\_loocv.py --model eegnet

```



For a limited test using selected patients:



```powershell

python scripts\\run\_dl\_loocv.py --model eegnet --patients chb01,chb02

```



The main result is written to:



```text

results\\per\_fold\_eegnet.csv

```



Training logs are written under:



```text

results\\training\_logs\\

```



Raw test probabilities are written under:



```text

results\\probabilities\\

```



\---



\# 13. Stage 1.5 — 0.5–25 Hz EEGNet Control Experiment



\## Purpose



Stage 1.5 tests whether information in the 25–50 Hz frequency range, including possible gamma-band or EMG-related information, contributes substantially to the EEGNet performance.



This experiment uses a separate cache so that the original Stage 1 cache is preserved.



\---



\## 13.1 Prepare the 0.5–25 Hz Cache



Run:



```powershell

python scripts\\prepare\_raw\_segments\_25hz.py

```



This creates the separate 0.5–25 Hz cache.



The Stage 1 cache should not be deleted.



\---



\## 13.2 Run Stage 1.5



Run:



```powershell

python scripts\\run\_dl\_loocv\_25hz.py

```



The expected result is:



```text

results\\per\_fold\_eegnet\_25hz.csv

```



Interpretation:



\* A substantial AUC reduction suggests that the 25–50 Hz range contributed to the observed performance.

\* AUC remaining high suggests that useful seizure-discriminative information can be learned from frequencies below 25 Hz.



\---



\# 14. Stage 2 — Per-Recording / Global Z-Score



\## Purpose



Stage 2 evaluates the effect of the normalization strategy.



Stage 1 applies per-segment, per-channel z-score normalization.



Stage 2 instead calculates channel statistics from the training pool and applies those training statistics to validation and test data.



This experiment reuses the original 0.5–50 Hz cache from Stage 1.



\---



\## 14.1 Run Stage 2



Run:



```powershell

python scripts\\run\_dl\_loocv\_perrec.py

```



The expected result is:



```text

results\\per\_fold\_eegnet\_perrec.csv

```



The original Stage 1 cache must remain available.



Interpretation:



\* A substantial reduction in AUC suggests that the normalization strategy contributed importantly to the Stage 1 performance.

\* Similar performance suggests that the model's performance is less dependent on per-segment amplitude normalization.



\---



\# 15. Stage 3 — Classical Random Forest with Per-Segment Z-Score



\## Purpose



Stage 3 is a normalization-control experiment.



It evaluates a classical Random Forest using features extracted from per-segment z-scored EEG segments.



The purpose is to investigate whether the apparent advantage of deep learning can be explained primarily by the normalization strategy rather than by model architecture.



Stage 3 reuses the original 0.5–50 Hz cache.



\---



\## 15.1 Run Stage 3



Run:



```powershell

python scripts\\run\_classical\_perseg.py

```



The expected result is:



```text

results\\per\_fold\_classical\_perseg.csv

```



Stage 3 is CPU-bound and does not require a GPU.



The implemented Random Forest uses features including:



\* relative Welch band powers;

\* Hjorth parameters;

\* line length;

\* mean absolute signal amplitude;

\* standard deviation.



The feature representation contains 198 features per segment.



\---



\# 16. Recommended Complete Execution Order



For a complete reproduction, follow this order:



\### Step 1 — Create the environment



```powershell

python -m venv .venv

```



\### Step 2 — Activate the environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### Step 3 — Install PyTorch



Install a CUDA-compatible PyTorch build for the target system.



\### Step 4 — Install the remaining dependencies



```powershell

python -m pip install -r requirements.txt

```



\### Step 5 — Verify the GPU



```powershell

python scripts\\verify\_gpu.py

```



\### Step 6 — Download CHB-MIT



Download the dataset separately from PhysioNet.



\### Step 7 — Configure `DATA\_ROOT`



Set the local dataset path in:



```text

src\\config.py

```



\### Step 8 — Build the main cache



```powershell

python scripts\\prepare\_raw\_segments.py

```



\### Step 9 — Run Stage 1



```powershell

python scripts\\run\_dl\_loocv.py --model eegnet

```



\### Step 10 — Run Stage 1.5



```powershell

python scripts\\prepare\_raw\_segments\_25hz.py

python scripts\\run\_dl\_loocv\_25hz.py

```



\### Step 11 — Run Stage 2



```powershell

python scripts\\run\_dl\_loocv\_perrec.py

```



\### Step 12 — Run Stage 3



```powershell

python scripts\\run\_classical\_perseg.py

```



\---



\# 17. Expected Main Results Files



After the corresponding experiments finish, the main CSV outputs are:



```text

results\\

├── per\_fold\_eegnet.csv

├── per\_fold\_eegnet\_25hz.csv

├── per\_fold\_eegnet\_perrec.csv

└── per\_fold\_classical\_perseg.csv

```



Additional generated files may include:



```text

results\\training\_logs\\

results\\probabilities\\

```



These are experiment outputs and are not source files required for running the pipeline.



\---



\# 18. Cache Locations



The main Stage 1 cache is controlled by the project configuration:



```text

RAW\_CACHE\_DIR

```



In the original experimental setup, the main cache was stored outside the repository.



Stage 1.5 uses a separate 0.5–25 Hz cache so that the original Stage 1 cache is preserved.



The cache files are large generated data products and are not included in the GitHub repository.



\---



\# 19. Important Data and Repository Restrictions



The following should remain outside the GitHub source repository:



```text

.venv\\

results\\

cache\_raw\\

cache\_raw\_25hz\\

```



The CHB-MIT dataset must also remain outside the repository.



These files/directories contain generated data, local environments, or experiment outputs and are not required as source files in the public code repository.



\---



\# 20. Reproducibility Principles



For reproduction of the reported experiments:



1\. Use the same 21-patient cohort.

2\. Use the same preprocessing configuration.

3\. Use 4-second windows with 50% overlap.

4\. Use the 18-channel bipolar montage.

5\. Use the same random seed (`42`).

6\. Preserve the Stage 1 cache when running Stage 2 and Stage 3.

7\. Use the separate 0.5–25 Hz cache for Stage 1.5.

8\. Keep the supplied source code and experiment configuration unchanged unless a local path must be configured for the dataset.

9\. Compare the generated per-fold CSV outputs with the reported experimental results.



\---



\# 21. Quick Command Reference



\## GPU verification



```powershell

python scripts\\verify\_gpu.py

```



\## Stage 1 cache



```powershell

python scripts\\prepare\_raw\_segments.py

```



\## Stage 1 EEGNet



```powershell

python scripts\\run\_dl\_loocv.py --model eegnet

```



\## Stage 1.5 cache



```powershell

python scripts\\prepare\_raw\_segments\_25hz.py

```



\## Stage 1.5 EEGNet



```powershell

python scripts\\run\_dl\_loocv\_25hz.py

```



\## Stage 2



```powershell

python scripts\\run\_dl\_loocv\_perrec.py

```



\## Stage 3



```powershell

python scripts\\run\_classical\_perseg.py

```



\---



\# 22. Final Note



This repository provides the source code and scripts used for the Deep Learning V3 experimental pipeline.



The dataset and generated caches are intentionally not distributed with the repository.



A complete reproduction therefore requires obtaining the CHB-MIT Scalp EEG dataset separately, configuring its local path, generating the required cache, and executing the experiment scripts described above.



