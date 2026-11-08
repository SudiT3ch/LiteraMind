"""SYNTHETIC FIXTURE CORPUS -- NOT REAL PUBLICATIONS.
Hand-written documents used only to exercise the pipeline offline (tests, smoke experiments).
Titles/authors/datasets are fictional. Results computed on this corpus say nothing about real-world performance."""
from __future__ import annotations

DOCS = {
 "syn_xai1": dict(title="[SYNTHETIC] Saliency Explanations for Chest X-ray Classifiers", year=2021, authors=["A. Fixture"], pages=[
  "Abstract\nWe propose a gradient-based saliency method to explain convolutional neural network predictions on chest X-ray images. The explanations help radiologists understand model decisions and improve trust in diagnosis.\n\nIntroduction\nMedical diagnosis systems based on deep learning are accurate but opaque. Explainable artificial intelligence can improve reliability and clinician trust.",
  "Methodology\nWe train a ResNet-50 convolutional neural network and generate Grad-CAM saliency maps. We evaluate on the SynthChest dataset containing 5000 images from a single hospital.\n\nResults\nThe model achieves an accuracy of 91% on SynthChest. Radiologists rated 78% of saliency maps as clinically useful.\n\nLimitations\nThe dataset comes from a single hospital and lacks external validation. Explanations were not evaluated quantitatively for faithfulness.\n\nFuture Work\nWe plan multi-centre validation and faithfulness metrics."]),
 "syn_xai2": dict(title="[SYNTHETIC] SHAP Feature Attribution for Tabular Clinical Risk Models", year=2022, authors=["B. Fixture"], pages=[
  "Abstract\nThis study applies SHAP feature attribution to gradient boosting models that predict patient risk from electronic health records. Interpretability supports clinical decision support.\n\nIntroduction\nClinical decision support requires transparent models. Trustworthy AI depends on explanations that clinicians can verify.",
  "Methodology\nWe train XGBoost models and compute SHAP values on the SynthEHR dataset of 12000 patient records from two clinics.\n\nResults\nThe model reaches an AUC of 0.87 on SynthEHR. SHAP explanations identified known risk factors in most cases.\n\nLimitations\nThe study relies on a single dataset and lacks external validation. Explanation stability across retraining was not examined. Clinician user studies were not conducted.\n\nConclusion\nSHAP offers practical interpretability for tabular clinical models."]),
 "syn_xai3": dict(title="[SYNTHETIC] Attention-Based Interpretability in Radiology Report Generation", year=2023, authors=["C. Fixture"], pages=[
  "Abstract\nWe present a transformer model with attention visualisation for radiology report generation, aiming to make automated diagnosis more interpretable.\n\nIntroduction\nModel interpretability is essential for clinical adoption of generative diagnosis systems.",
  "Methodology\nA vision transformer encoder is paired with a language decoder. Experiments use the SynthChest dataset and the SynthReports dataset.\n\nResults\nThe model attains a BLEU score of 0.31 and radiologists preferred the attention-annotated reports in 64% of cases.\n\nLimitations\nAttention weights may not reflect true model reasoning. The evaluation lacks external validation and clinician user studies were limited in size.\n\nFuture Work\nFuture work should test faithfulness of attention explanations."]),
 "syn_xai4": dict(title="[SYNTHETIC] Counterfactual Explanations for Diabetes Diagnosis Models", year=2022, authors=["D. Fixture"], pages=[
  "Abstract\nCounterfactual explanations are generated for diabetes diagnosis classifiers to show clinicians which patient features would change a prediction.\n\nIntroduction\nExplainable AI can improve reliability in medical diagnosis by exposing decision boundaries.",
  "Methodology\nWe use a random forest classifier and a counterfactual search algorithm on the SynthDiabetes dataset of 800 patients.\n\nResults\nThe random forest reaches accuracy of 84% on SynthDiabetes. Counterfactuals were valid in 92% of tested cases.\n\nLimitations\nThe SynthDiabetes dataset is small with only 800 patients. Counterfactuals may suggest clinically unrealistic changes. No user study was performed.\n\nFuture Work\nLarger datasets and clinician evaluation are needed."]),
 "syn_fl1": dict(title="[SYNTHETIC] Federated Averaging for Privacy-Preserving Hospital Models", year=2021, authors=["E. Fixture"], pages=[
  "Abstract\nWe study federated learning across hospitals so that patient data never leaves local sites, using federated averaging to train a shared diagnosis model.\n\nIntroduction\nPrivacy regulations restrict sharing of medical records. Federated learning enables collaborative training.",
  "Methodology\nWe simulate five hospitals with the SynthEHR dataset partitioned by clinic and train with federated averaging.\n\nResults\nThe federated model reaches an AUC of 0.84 compared with 0.87 for centralised training on SynthEHR.\n\nLimitations\nCommunication cost is high and performance degrades with non-identical data across sites. Privacy guarantees were not formally analysed."]),
 "syn_fl2": dict(title="[SYNTHETIC] Differential Privacy in Federated Medical Imaging", year=2023, authors=["F. Fixture"], pages=[
  "Abstract\nThis work adds differential privacy to federated training for medical image classification and measures the privacy-utility trade-off.\n\nIntroduction\nFederated learning alone does not guarantee privacy; differential privacy provides formal bounds.",
  "Methodology\nWe train a ResNet-18 with DP-SGD in a federated setting on the SynthChest dataset split across eight simulated sites.\n\nResults\nAccuracy drops from 90% to 83% on SynthChest as the privacy budget tightens.\n\nLimitations\nAccuracy degrades under strong privacy budgets. Experiments use simulated sites and lack real multi-hospital deployment. Communication cost is high."]),
}

# Ground-truth relevance for retrieval evaluation: query -> set of relevant paper ids (author-assigned for the fixture).
QUERIES = {
 "How can explainable AI improve reliability of medical diagnosis?": {"syn_xai1", "syn_xai2", "syn_xai3", "syn_xai4"},
 "saliency explanations chest X-ray radiologists": {"syn_xai1", "syn_xai3"},
 "federated learning privacy hospitals": {"syn_fl1", "syn_fl2"},
 "SHAP feature attribution clinical risk": {"syn_xai2"},
 "differential privacy medical imaging accuracy drop": {"syn_fl2"},
 "counterfactual explanations diabetes": {"syn_xai4"},
}

# Author-assigned gold labels for the SYNTHETIC corpus (written together with the documents -> only a pipeline sanity check).
GOLD_COMPARABLE = {("AUC", "SynthEHR"): {"syn_xai2", "syn_fl1"}, ("Accuracy", "SynthChest"): {"syn_xai1", "syn_fl2"}}
GOLD_GAP_THEMES = {
    "external_validation": {"syn_xai1", "syn_xai2", "syn_xai3", "syn_fl2"},
    "user_study": {"syn_xai2", "syn_xai3", "syn_xai4"},
    "cost": {"syn_fl1", "syn_fl2"},
    "small_data": {"syn_xai4"},
}
