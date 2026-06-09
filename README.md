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
**Data Source** 
Literature citations where the data originates: 
https://drive.google.com/file/d/1cHMTxjyj0Bh8gJpWKe3aP9n0DZ9PKlBa/view?usp=sharing
**Recording Device**
Neuroscan, 64 channels, sampling rate 1000Hz
**Original Purpose**
An Event-Related Potential Study on Function Graphs and Equation Transformation Abilities in Mathematically Gifted Senior High School Students





