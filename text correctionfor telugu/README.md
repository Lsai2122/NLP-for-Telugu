# Automatic Text Correction for Low-Resource Languages (Telugu)

**IIIT Nagpur, Dept of CSE, 5th Semester Mini Project**

## Team Members
- Leela Venkata Sai Nerella (BT24CSA008)
- Parthasaradhi Guda (BT24CSA061)
- Dineswara Kumar Mangali (BT24CSA062)

## Supervisors
- Ms. Dipti Pawade
- Ms. Jaya Pathak

## Problem Statement
Automatic text correction for low-resource languages like Telugu presents unique challenges due to complex morphology, rich vocabulary, and the lack of large-scale annotated datasets. This project aims to develop a robust text correction system tailored for Telugu, addressing common errors such as typographical mistakes, matra confusions, and phonetically similar consonant substitutions.

## Tech Stack
![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd telugu-text-correction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the Streamlit application for text correction:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
telugu-text-correction/
│
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
├── src/
│   ├── correction/
│   ├── error_detection/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── utils/
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Methodology Overview

The system employs a multi-step pipeline for text correction:
1.  **Preprocessing**: Normalization, script detection, and transliteration (Tenglish to Telugu).
2.  **Error Detection**: A hybrid approach combining dictionary lookups, rule-based checks, and neural Masked Language Modeling (MLM).
3.  **Correction**: Generating candidates via edit distance, rule-based mappings, and Transformer-based models (IndicBART/mT5).
4.  **Evaluation**: Rigorous testing using established metrics.

## Evaluation Metrics
- Word Error Rate (WER)
- Character Error Rate (CER)
- BLEU Score
- Precision, Recall, and F1-Score

## References
- Key papers and resources related to Indic NLP and Telugu text processing.

## License
MIT License
