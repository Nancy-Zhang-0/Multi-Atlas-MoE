# Multi-Atlas-MoE

This repository trains mixture-of-experts (MoE) models on multi-atlas brain structural/functional connectivity from ADNI subjects to study Mild Cognitive Impairment (MCI). 

![Multi-Atlas MoE overview](Multi-Atlas-MoE/Model/MultiAtlas.png)

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/Nancy-Zhang-0/Multi-Atlas-MoE.git
   cd Multi-Atlas-MoE
   ```

2. **Create an environment and install requirements**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare data**
   Place the cleaned ADNI JSON manifests and the referenced `.npy` connectivity matrices in the project directory. Each JSON entry should include keys such as `AAL_SC_path`, `AAL_FC_path`, `3Hinge_SC_path`, etc.

## Usage

### Training
```bash
python train_moe_gcn.py \
  --train-json ADNI_train_no_mmse_clean_complete.json \
  --cv-folds 5 \
  --epochs 128 \
  --batch-size 8 \
  --device cuda \
  --save-model-dir models/moe_cv \
  --save-training-artefacts logs/moe_cv
```
- The GCN MoE pipeline performs 5-fold stratified cross-validation.
- Use `--atlases AAL,HOA,...` to restrict the atlas set if needed.
- Add `--use-default-atlas-config` for pre-tuned attention settings, or pass a custom JSON via `--atlas-config`.

## Citation

If this repository contributes to your research, please cite it and acknowledge the ADNI dataset in your publication.

## License

This project is distributed under the MIT License.

## Contact

Questions or collaboration ideas? Open an issue or reach out to [Nancy-Zhang-0](https://github.com/Nancy-Zhang-0).

