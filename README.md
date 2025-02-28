# 🎼 Classical Music Generation using Neural Networks

## 🎻 Project Overview
This project explores the intersection of art and technology by using Neural Network models to generate classical music compositions. The goal is to uncover the predictability of musical patterns in classical music and craft authentic compositions that resonate with the rich legacy of classical music. 

Leveraging a dataset of classical compositions in MIDI format, we employ advanced techniques such as Long Short-Term Memory (LSTM) networks to generate original music sequences, pushing the boundaries of AI-powered music generation.

---

## 🎯 Objectives
- **Data Preparation**: Extract key musical elements from MIDI files, such as pitch, velocity, start time, and end time.
- **Model Development**: Train LSTM models to learn temporal dependencies within musical sequences.
- **Music Generation**: Use the trained model to predict and generate the next note in a sequence, mimicking techniques similar to next-token generation used in language models.
- **Evaluation and Refinement**: Experiment with various hyperparameters to optimize model performance and produce coherent compositions.

---

## 🔀 Project Flow

The project followed a structured workflow:

1. **Data Collection**:
   - Loaded classical music compositions in MIDI format, focusing on a selected composer.
2. **Data Preprocessing**:
   - Used **PrettyMIDI** to extract musical elements such as:
     - **Pitch**: Frequency of notes.
     - **Velocity**: Intensity of notes.
     - **Start & End times**: Timing of each note in the composition.
   - Explored multiple extraction methods, including:
     - **Piano rolls**: Representing MIDI data in a grid format.
     - **Waveform analysis**: To capture additional musical features.
3. **Sequence Representation**:
   - Transformed the extracted data into sequential representations, suitable for feeding into neural networks.
4. **Model Setup**:
   - Defined sequence length to balance temporal context.
   - Experimented with hyperparameters, including embedding dimensions and batch size.
   - Explored an **initial regression approach** for pitch prediction, later pivoting to next-note generation.
5. **Training**:
   - Trained LSTM models to learn patterns in note sequences and predict the most probable next note.
6. **Music Generation**:
   - Used the trained models to autonomously generate new musical sequences.
7. **Output**:
   - Converted the generated note sequences back into MIDI format, producing AI-composed classical pieces.

---

## 🛠️ Technologies Used

- **Python**: For data processing and model development
- **PrettyMIDI**: MIDI file parsing and musical element extraction
- **TensorFlow & Keras**: Neural network modeling (LSTM)
- **Jupyter Notebooks**: Exploration and iterative development
- **MIDI Files**: Dataset of classical music compositions
- **Matplotlib & Seaborn**: Visualization of musical patterns

---

## 🧪 Key Contributions

### 🎼 Data Preparation
- Extracted important musical elements from MIDI files using **PrettyMIDI**.
- Processed attributes like **pitch**, **velocity**, **start time**, and **end time** to create meaningful musical sequences.
- Explored alternative data extraction techniques:
  - **Piano roll representations** to visually map notes over time.
  - **Waveform analysis** to capture richer musical nuances.
- Focused on a **next-note generation approach** — inspired by large language models — to predict the next note in a sequence.

### 🏗️ Model Development
- Defined **sequence length** to balance musical context and model complexity.
- Experimented with **hyperparameters**:
  - **Sequence length**: Temporal span of input sequences.
  - **Embedding dimensions**: Capturing semantic relationships between notes.
- Explored an **initial regression approach** for pitch prediction — an important learning step despite its limitations — before pivoting to next-note generation using LSTM.

---

## 🚀 Running the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Classical-Music-Generation.git
