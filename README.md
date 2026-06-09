# EEG-Based Stage Classification in Mathematical Problem Solving

## Introduction
**What specific human information processing function are you investigating or testing? (Examples: working memory encoding, selective attention, visual processing, semantic processing, or autonomic reactions to stress). Provide a concise theoretical background and clearly state your team’s core objective or hypothesis.**

My final project explores the differences in neural patterns across different modes of high-order cognitive tasks—specifically, how students comprehend the translational relationship between mathematical graphs and equations. While previous research has investigated the event-related potential (ERP) patterns underlying graph-equation relationships, traditional ERP analysis relies heavily on averaging EEG signal amplitudes across numerous trials to isolate these components. Consequently, this project investigates whether it is possible to decode EEG patterns directly from raw, single-trial signals without conventional averaging. To achieve this, I compared a machine learning method (Random Forest) with a deep learning architecture (EEGNet) to classify EEG segments corresponding to the graph and equation comprehension stages.

**Are you planning to utilize specific EEG indices or HRV metrics? If so, clearly identify what they are.**

For the random forest classifier, several features were extracted from the EEG segments during both the graph and equation comprehension stages:
1. Absolute and relative band power: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz)
2. Inter-band power ratios: theta/alpha, delta/alpha, theta/beta, and alpha/beta
3. Amplitude asymmetry and FFT coherence: calculated for channel pairs F3/F4, C3/C4, P3/P4, F7/F8, T7/T8, and O1/O2
4. Hjorth parameters: activity, mobility, and complexity

**Please explain the explicit association between your chosen cognitive topic and these physiological measures. Why are these specific metrics valid indicators for the information processing function you are dealing with?**

1. Absolute band power was computed for each electrode channel across five canonical frequency bands—delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz)—yielding N_ch × 5 features per trial, where N_ch denotes the number of EEG channels. This feature group captures the raw spectral energy distribution across cortical regions and frequency ranges implicated in visual processing, working memory, and executive function during mathematical problem solving. Relative band power was subsequently derived by normalising each band's absolute power by the total broadband power within the same channel and trial, producing a complementary set of N_ch × 5 scale-invariant spectral features that are less sensitive to inter-subject amplitude variability.
2. Inter-band power ratios were computed as four channel-averaged scalar quantities: theta/alpha, delta/alpha, theta/beta, and alpha/beta. These ratios reflect the relative dominance of slower versus faster oscillatory rhythms and have established sensitivity to cognitive load, attentional resource allocation, and working memory engagement —all of which are expected to differ meaningfully between passive graph reading (S1) and active representational transformation (S2).
3. Amplitude asymmetry was quantified as the log-ratio of band power between six pairs of homologous left–right electrode pairs: F3/F4, C3/C4, P3/P4, F7/F8, T7/T8, and O1/O2.  Positive values indicate right-hemisphere dominance; negative values indicate left-hemisphere dominance within a given band. This feature group is particularly relevant to the present paradigm, as function graph-to-equation transformation has been shown to produce lateralisation effects in parieto-occipital regions, with the right posterior cortex exhibiting greater activation during S2.
4. FFT-based spectral coherence was estimated between the same six electrode pairs across the five frequency bands, yielding an additional 30 features (6 pairs × 5 bands). Coherence quantifies the degree of linear synchrony between spatially separated neural populations and serves as an index of functional connectivity between frontal and posterior cortical regions. Such fronto-posterior coupling is associated with the integration of visual-perceptual and executive processing resources during complex mathematical cognition.
5. Hjorth parameters—activity, mobility, and complexity—were computed per channel, contributing 3 × N_ch features. Given that gifted students have been reported to exhibit distinct patterns of cortical activity during representational transformation tasks, Hjorth parameters provide a complementary time-domain characterisation of these differences without requiring explicit frequency decomposition.

## Data Source & Technical Parameters
**Data Source:** 

Literature citations where the data originates: 
https://drive.google.com/file/d/1cHMTxjyj0Bh8gJpWKe3aP9n0DZ9PKlBa/view?usp=sharing 

**Recording Device:**
Neuroscan, 64 channels, sampling rate 1000Hz

**Original Purpose:**
An Event-Related Potential Study on Function Graphs and Equation Transformation Abilities in Mathematically Gifted Senior High School Students


<img width="1252" height="402" alt="image" src="https://github.com/user-attachments/assets/835127f8-9ae2-4d07-bf49-b9a5023f6696" />

The test consists of 60 items, comprising 30 items with correct answers and 30 items with incorrect answers. All items were designed with a two-stage structure: the first stage required students to read a function graph (S1), while the second stage presented an equation and asked students to judge, based on the function graph from the first stage, whether the given equation was correct (S2). The function graphs and equations included in the test items cover mathematical content that high school students have studied. The trial sequence began with a 1000 ms fixation period, followed by a 2000 ms function graph reading stage (S1). After a brief 1000 ms inter-stimulus interval, the task proceeded to a 5000 ms function graph and equation transformation stage (S2), and concluded with a final 1000 ms inter-stimulus interval.

Each trial was cropped for 2 seconds from the onset of Stage 1 and Stage 2. The segments were then labeled into two different groups, designated for binary classification using EEGNet and a Random Forest Classifier.

## Data Preprocessing & Quality Control
**Visual Inspection & Artifact Identification: Include clear screenshots of your
raw data identifying specific biological or environmental artifacts (such as eye
blinks, muscle activity/EMG, sweat/baseline drift, or 50/60 Hz line noise).**
<img width="1266" height="784" alt="image" src="https://github.com/user-attachments/assets/c8cb4035-80fb-417b-bd06-87e4ce1bdf9e" />


**Spectral & Time-Series Proof: Provide clear plots showing both the Time-Series
Signal and the Power Spectral Density (PSD) before and after your preprocessing
pipeline (e.g., after applying bandpass filters and artifact removal techniques).**
<img width="2084" height="1330" alt="image" src="https://github.com/user-attachments/assets/83875430-b658-44e2-b1e4-abdab2b7466d" />

**Initial Input:** Raw EEG data containing 67 channels sampled at a frequency of 1000 Hz.

**Step 1:** Drop Trigger Channel: The trigger channel is removed from the dataset, leaving only the signal channels.

**Step 2:** Average Re-reference: The data is re-referenced to the average of all channels to reduce global noise.

**Step 3:** Bandpass Filter: A Finite Impulse Response (FIR) filter is applied between 1 – 50 Hz. This removes DC drift (low frequencies) and electromyogram/muscle noise (high frequencies).

**Step 4:** Notch Filter: A notch filter is applied specifically at 60 Hz to eliminate electrical interference from the powerline.

**Step 5:** Independent Component Analysis (ICA): The FastICA algorithm is run to extract 20 components. The identified Electrooculogram (EOG/eye blink) components are excluded and removed from the data.

**Final Output:** Cleaned EEG data, now reduced to 66 channels.

## End-to-End Analysis Pipeline Demo Video

## Analytical Results & Interpretation

**Random Forest Classification Results**
<img width="1015" height="734" alt="image" src="https://github.com/user-attachments/assets/9ba624ea-16da-42d3-a1e5-33ced280d97b" />

<img width="1485" height="1243" alt="image" src="https://github.com/user-attachments/assets/7bb09e82-70cd-4e74-b1ab-23e3d736a317" />



**EEGNet Classification Results**
<img width="966" height="714" alt="image" src="https://github.com/user-attachments/assets/fe1db3bd-8b59-4f45-bc07-c5523ad7b84c" />

**Interpretation**








