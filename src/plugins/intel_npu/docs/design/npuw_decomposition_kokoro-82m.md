# Decomposition design in NPUW for kokoro-82m

## Goal

- Support kokoro-82m inside NPU plugin with static shape input (fixed number of tokens).
- Achieve reasonable performance compared to CPU & GPU solutions.

## Problem statement

The NPU Plugin lacks support for dynamic shapes in the kokoro-82m scenario. There are currently two approaches implemented and used for dynamic models:

1. Upper bound dynamism - which is not applicable for the kokoro scenario because the upper bound size (and memory allocation) for the worst-case scenario is several times larger than the real data. See ["Bounded Dynamism"](#b-bounded-dynamism) section.
2. NPUW decomposition - currently focused on LLM models (decomposition into prefill and kvcache) and the Whisper model.

## Context

### Kokoro-82m model

Kokoro-82m model is composition of different models (bert, for example), but in general can be separated into two parts by using layer which introduce dynamism to this model - torch.repeat_interleave operation, which performs a "scatter/gather" or "expansion" based on the *values* inside `pred_dur`.

- `Text Encoder` $\to$ `Duration Predictor` $\to$ **[The Bridge]** $\to$ `Decoder`.
- Input of the model can be converted to static, but bridge introduce global data-dependent dynamism.

```mermaid
graph LR
    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef token fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef bridge fill:#e0f2f1,stroke:#00695c,stroke-width:2px,stroke-dasharray: 5 5;
    classDef frame fill:#f3e5f5,stroke:#880e4f,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph Inputs [Inputs]
        direction TB
        Tokens(Text Tokens):::input
        Style(Style Vector):::input
    end

    subgraph TokenDomain [Token Domain]
        direction TB
        BERT[BERT & Prosody Encoder]:::token
        TextEnc[Content Encoder]:::token
        DurPred[Duration Predictor]:::token
    end

    subgraph Bridge [The Bridge]
        Align((Alignment / Expansion<br>repeat_interleave)):::bridge
    end

    subgraph FrameDomain [Frame Domain]
        direction TB
        F0N[Pitch & Noise Predictor]:::frame
        Decoder[Decoder]:::frame
    end

    Audio(Audio Waveform):::output

    %% Connections
    Tokens --> BERT
    Tokens --> TextEnc
    Style --> BERT
    Style --> Decoder

    BERT --> DurPred
    DurPred -- "How long?" --> Align
  
    BERT -- "Prosody Features<br>(How to say)" --> Align
    TextEnc -- "Content Features<br>(What to say)" --> Align

    Align -- "Aligned Prosody" --> F0N
    Align -- "Aligned Content" --> Decoder
  
    F0N -- "F0, N" --> Decoder
    Decoder --> Audio

```

### Repeat interleave layer 

Input of repeat interleave (pred_dur) is a vector of integers indicating how many time frames each input token lasts. 

```mermaid
graph LR
    subgraph INPUT ["Input Tensor"]
    direction TB
    I1[Phoneme: A]
    I2[Phoneme: B]
    I3[Phoneme: C]
    end

    subgraph REPEATS ["Repeats Tensor (Durations)"]
    direction TB
    R1["2"]
    R2["1"]
    R3["3"]
    end

    subgraph OUTPUT ["Output (Dynamic Size)"]
    direction TB
    O1[A] --- O2[A] --- O3[B] --- O4[C] --- O5[C] --- O6[C]
    end

    I1 -->|"repeats"| R1 --> O1 & O2
    I2 -->|"repeats"| R2 --> O3
    I3 -->|"repeats"| R3 --> O4 & O5 & O6

    style REPEATS fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

- tokens:     [CLS], today, is, rainy, ., [SEP] (6 tokens, can be static)
- pred_dur: [0, 2, 1, 3, 1, 0] (N = 6)
- result [1, 1, 2, 3, 3, 3, 4] - dynamic
  - upper bound = number of tokens * max phoneme length  (real cases 0-50 frames)
  - upper bound size = 6 * 10 = 60, real size = 7

When it's represented as single operation inside Pytorch, during import of OpenVINO it's transformed to sequence of different operation like `Range`, `Tile` to reproduce the logic using available inside OpenVINO operations.

#### Repeat interleave + alignment

Alignment is conversion from token domain to time domain.

Operation: Y=F×A

F - Feature matrix \[Features, Tokens\] for two tokens (A & B)

$$
F=\begin{bmatrix}A_{1}&B_{1} \\\ A_{2}&B_{2} \\\ A_{3}&B_{3}\end{bmatrix}
$$

A - Alignment Matrix \[Tokens, Time\] (Sparse, mostly zeros)

$$
A=\begin{bmatrix} 1&1&0 \\\ 0&0&1  \end{bmatrix}
$$

Y - features in time domain \[Features, Time\]

$$
Y= \begin{bmatrix} A_{1}&A_{1}&B_{1} \\\ A_{2}&A_{2}&B_{2} \\\ A_{3}&A_{3}&B_{3} \end{bmatrix}
$$

## Design

- **Concept**:
  Use the location where dynamism is introduced as the cutting point (`repeat_interleave` layer) to split the original model into two models. The first `Predictor` (Model A) will be responsible only for duration prediction and generating two feature matrices (Prosody Features and Content Features). For the second `Decoder` (Model B), replace the dynamic dimension introduced inside `repeat_interleave` with a fixed-size block (small size, 100-200 timeframes to avoid long compilation), thereby converting the model into a static version. To connect both models, generate multiple fixed-size aligned feature matrices (see [Repeat interleave alignment](#repeat-interleave--alignment)) inside the loop (create fixed-size matrix -> run model b -> gather result). The loop will be executed on host side (CPU).

- **Roadblocks**:
  1. **Context Continuity**: Audio generation is sensitive. Splitting the graph often creates "clicks" or "fading" at the boundaries. We need complex overlap handling (Host complexity).
  2. **Compilation Overhead**: The second part (Decoder) is massive (STFT/ISTFT) and takes a long time to compile.
- **Limitations**:
  1. **Static input**: Static input required in order to do decomposition into two static parts.
  2. **Duplication for other inputs**: Since model contain also style (`ref_s`) and speed inputs, there will be some duplication between two models. These inputs will be used for both model.

### Decomposition

#### Predictor

**How to cut**: Original model already contain predictor as one of the model outputs. All we need to do to create copy of original model with only `pred_dur` (predicted duration) output.

#### Decoder

**How to cut**: Inside the kokoro-82m model, there is a large subgraph responsible for the repeat interleave layer and creating the alignment matrix (see [Repeat interleave layer](#repeat-interleave-layer)). Since the responsibility of creating the aligned matrix in this decomposed pipeline will be on the host side, we need to create a model that accepts both aligned matrices as input. The diagram below represents the location in the original model where this alignment is performed. To create the static decoder model, we have to find the location of both MatMul operations, use their outputs as inputs for the new model, and replace the dynamic `Time` dimension with a fixed block size.

```mermaid
graph TD
    subgraph Predictor Output
        Unsqueeze["Unsqueeze_2"]:::op
        
        Feat1["Feature Tensor 1<br>1 x 640 x tokens"]:::tensor
        Feat2["Feature Tensor 2<br>1 x 512 x tokens"]:::tensor
        
        AlignMatrix["pred_aln_trg<br>1 x tokens x Time"]:::tensor
        
        Unsqueeze --> AlignMatrix
    end

    subgraph The_Bridge ["Alignment / Cutting Point"]
        direction TB
        MM1["MatMul_1<br>in1: 1 × 640 × tokens<br>in2: 1 × tokens x Time<br>out: 1 x 640 x Time"]:::cut
        MM2["MatMul<br>in1: 1 × 512 × tokens<br>in2: 1 × tokens x Time<br>out: 1 x 512 x Time"]:::cut
    end
    
    %% Connections
    AlignMatrix --> MM1
    AlignMatrix --> MM2
    
    Feat1 --> MM1
    Feat2 --> MM2
```

### Manual alignment + loop execution

The host application will implement the alignment logic (equivalent to `torch.repeat_interleave` followed by matrix multiplication)
First, create `idx` array which can be used in Gather operation to extract features vectors from matrix.  
```
pred_dur        idx
[1, 2, 3, 1] -> [0, 1, 1, 2, 2, 2, 3]
```
On this stage we moved from token to time domain.  

Then, in loop, using fixed block size (limiting maximum time frames processed at once), we need to create aligned in time feature matrix
```
Features (L=3)      Indices (idx)        Destination (block=5)  
[ A | B | C ]   <-- [0, 0, 1, 2, 2] --> [ A | A | B | C | C ]
```
Run model b using it and process audio output (add result / handle padding).

## Alternatives considered

### A. Static Compilation

- **Status**: Impossible.
- **Reason**: We cannot pre-calculate memory offsets or execution schedules because the tensor sizes are literally unknown.

### B. Bounded Dynamism

- **Concept**: Set a maximum limit and pad everything.
- **Problem**: "Shape Explosion".
  - To be safe, Upper Bound = `Input_Length` $\times$ `Max_Possible_Duration_Per_Token`.
  - *Example*: 500 tokens $\times$ 50 frames/token = **25,000 frames** allocated.
  - *Reality*: Average duration is ~5 frames. Actual needed = **2,500 frames**.
  - **Result**: 10x memory waste and massive compute waste processing zeros.
- **Compiler Issue**: Propagating these loose bounds through `Range` and `Tile` ops is complex and results in inefficient kernels.
