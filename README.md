# GenCam

## 🔍 Project Overview

**GenCam** is an intelligent visual scene analysis system that integrates traditional Image Signal Processing (ISP) techniques with cutting-edge Generative AI models. The system takes input from camera feeds or image files, enhances them using ISP-inspired optimization (e.g., exposure, contrast, noise reduction), and then generates natural language descriptions using vision-language models like BLIP or CLIP.

This project bridges embedded camera system knowledge with AI-based scene understanding — making it ideal for research in smart imaging, autonomous vision systems, or human-AI visual interpretation.

---

## 🚀 Features

- **ISP Auto-Tuning:** Applies grid-based optimization over brightness, contrast, and sharpness.
- **Generative Captioning:** Uses BLIP (Salesforce/blip-image-captioning-base) to generate accurate scene captions.
- **Reference Caption Scoring:** Compares AI captions to human captions (BLEU, BERTScore).
- **Batch Processing:** Scripts for auto-tuning across datasets and storing results.
- **Streamlit App:** Interactive UI for uploading images, tuning ISP, and generating captions.
- **COCO Dataset Integration:** Utilities for working with COCO val2017 images and annotations.

---

## 🏗️ Tech Stack

- **Python 3.8+**
- **Jupyter Notebook** (for experimentation)
- **Streamlit** (interactive web app)
- **PyTorch**
- **Transformers (HuggingFace)**
- **PIL/Pillow**
- **NLTK, evaluate** (BLEU, BERTScore metrics)
- **COCO dataset**

---

## 🖥️ Usage

### 1. Setup

```bash
git clone https://github.com/NaveenkumarBisai/GenCam.git
cd GenCam
pip install -r requirements.txt
```

Download the required NLTK resources:

```bash
python download.py
```

Ensure you have the COCO val2017 dataset in `data/coco/val2017` and the corresponding captions JSON in `data/coco/annotations/`.

### 2. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

- Upload a `.jpg`, `.jpeg`, or `.png` image.
- Optionally enter a reference (human) caption.
- The app will auto-tune ISP parameters and generate the best caption.

### 3. Run Batch Auto-Tuning

```bash
python scripts/auto_tune_batch.py
```

- Processes a batch of COCO images, auto-tunes ISP, and saves results in CSV.

### 4. Evaluate Variants

```bash
python scripts/evaluate_variants.py
```

---

## 📦 Repository Structure

```
GenCam/
│
├── streamlit_app.py            # Main web application
├── scripts/                    # Batch and evaluation scripts
│   ├── auto_tune_isp.py
│   ├── auto_tune_batch.py
│   └── evaluate_variants.py
├── modules/
│   ├── genai_captioning/       # BLIP-based captioning module
│   ├── isp/                    # ISP parameter transforms
│   └── eval/                   # Scoring (BLEU, BERTScore)
├── utils/                      # COCO loader and utilities
├── data/                       # (Place COCO val2017 images/annotations here)
├── results/                    # Output, logs, and CSVs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔗 References

- [BLIP: Bootstrapped Language Image Pretraining](https://github.com/salesforce/BLIP)
- [COCO Dataset](https://cocodataset.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

- [Naveen Kumar Bisai](https://github.com/NaveenkumarBisai)

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📢 TODO / Suggestions

- [ ] Add sample images and output captions to `README.md` for quick demo.
- [ ] Add Dockerfile for reproducible environment.
- [ ] Add Example API usage for integration in other projects.
- [ ] Add instructions for using other vision-language models (e.g., CLIP).