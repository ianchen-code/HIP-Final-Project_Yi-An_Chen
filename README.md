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


