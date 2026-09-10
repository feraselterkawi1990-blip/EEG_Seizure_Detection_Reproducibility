# EEG Seizure Detection — LOOCV Pipeline (v2)

Pipeline لتنفيذ Leave-One-Patient-Out Cross-Validation على قاعدة CHB-MIT
لرسالة "When Cross-Patient Validation Reveals the True Limits of Classical
Machine Learning for EEG Seizure Detection".

**هذه نسخة v2 (مايو 2026)**. للتغييرات الكاملة عن النسخة الأولى، انظر
`CHANGELOG.md`.

---

## ما الجديد في v2

- **XGBoost** أُضيف كنموذج ثالث رئيسي (issue #20).
- **Threshold tuning موحَّد** لكل النماذج عبر Youden's J على نفس validation
  set (issue #1، #2).
- **اختبارات إحصائية حقيقية**: Wilcoxon signed-rank مع Holm correction +
  bootstrap 95% CIs (issues #21، #22).
- **تحليل آلي للانهيار**: MMD + Wasserstein + Mahalanobis لكل مريض،
  مع Spearman correlations مع الـ AUC (issue #23).
- **كل المعلمات** مُنقولة إلى `config.py` (issue #4) — كما تَعِد §3.6.3
  من الأطروحة.
- **توثيق المرضى** مُصحَّح (issue #3): chb21 = إعادة تسجيل لـ chb01
  (وليس chb24)، 21 مريضاً مستخدماً.
- **معلمة ICA ميتة** (`ICA_VARIANCE_THRESHOLD`) أُزيلت، استُبدلت بـ
  `ICA_FRONTAL_ASYMMETRY_THRESHOLD` و `ICA_FRONTAL_PREFIXES` تطابقان
  ما يفعله `preprocessing.py` فعلاً (issue #5).

---

## 1. متطلبات النظام

- Python 3.10+
- 16 GB RAM كحد أدنى (32 GB موصى به)
- 50 GB مساحة على القرص (للبيانات + الكاش)
- GPU (اختياري، لتسريع LSTM/CNN فقط)

---

## 2. التثبيت

```bash
cd seizure_pipeline_v2
python -m venv venv
source venv/bin/activate           # على ويندوز: venv\Scripts\activate
pip install -r requirements.txt
```

XGBoost مُدرج بالفعل في `requirements.txt`. TensorFlow اختياري —
أزِل التعليق إن أردتَ تشغيل LSTM/CNN.

---

## 3. الإعداد

### الطريقة 1 (سريع): تعديل `config.py`
```python
CHB_MIT_PATH = Path(r"/your/path/to/chbmit")
```

### الطريقة 2 (موصى به للنشر): متغيّر بيئة
```bash
export CHBMIT_PATH=/your/path/to/chbmit       # Linux/Mac
$env:CHBMIT_PATH = "C:\path\to\chbmit"        # PowerShell
```

---

## 4. التشغيل

### اختبار سريع (مريض واحد، ~10 دقائق)
```bash
python scripts/run_loocv.py --smoke
```

### تشغيل كامل (RF + SVM + XGB، ~5-7 ساعات على CPU)
```bash
python scripts/run_loocv.py
```

### مع تحليل توزيع البيانات (مدّة إضافية ~30 دقيقة)
```bash
python scripts/run_loocv.py --shift
```

### استخدام كاش موجود مسبقاً (يتخطى مرحلة بناء الكاش)
```bash
python scripts/run_loocv.py --skip-cache
```

### إعادة الاختبارات الإحصائية فقط (من نتائج موجودة)
```bash
python scripts/run_statistical_tests.py
```

### إضافة LSTM/CNN
في `config.py`:
```python
MODELS_TO_RUN = ["RF", "SVM", "XGB", "CNN", "LSTM"]
```
ثم:
```bash
pip install tensorflow
python scripts/run_loocv.py --skip-cache
```

---

## 5. الزمن المتوقع (Ryzen 9 7940HS، CPU فقط)

| المرحلة | الزمن |
|---------|-------|
| بناء الكاش (21 مريضاً، مرة واحدة) | 4-5 ساعات |
| LOOCV — RF + SVM + XGB | 4-6 ساعات |
| LOOCV — مع CNN فقط (إضافة) | +50 ساعة |
| LOOCV — مع LSTM فقط (إضافة) | +100-200 ساعة |
| الاختبارات الإحصائية | < 1 دقيقة |
| تحليل توزيع البيانات | 20-40 دقيقة |

نصيحة: ابنِ الكاش مرة واحدة، ثم استخدم `--skip-cache` لكل التجارب.

---

## 6. المخرجات

```
results/
├── cache/
│   └── chb01.npz, chb02.npz, ...           # الكاش (لا يتغيّر)
├── tables/
│   ├── loocv_per_fold.csv                  # كل صف = (model, patient)
│   ├── loocv_summary.csv                   # mean ± std لكل model
│   ├── thesis_headline_table.csv           # الجدول الرئيسي
│   ├── bootstrap_ci_auc.csv                # NEW: CIs لكل model
│   ├── bootstrap_ci_sensitivity.csv        # NEW
│   ├── bootstrap_ci_specificity.csv        # NEW
│   ├── wilcoxon_auc_holm.csv               # NEW: مقارنات بين models
│   └── distribution_shift.csv              # NEW (مع --shift)
├── figures/
│   ├── loocv_boxplots.png                  # توزيع الأداء
│   ├── per_patient_accuracy.png            # دقة كل مريض
│   ├── aggregated_confusion.png            # confusion matrices
│   ├── catastrophic_collapse.png           # NEW: sens vs AUC
│   └── distribution_shift.png              # NEW (مع --shift)
└── logs/
    └── run_loocv.log
```

---

## 7. هيكل الكود (v2)

```
seizure_pipeline_v2/
├── config.py                       # كل الإعدادات (single source of truth)
├── requirements.txt
├── README.md                       # هذا الملف
├── CHANGELOG.md                    # شرح كل تغيير عن v1
├── src/
│   ├── data_loader.py              # قراءة EDF + ملفات الـ summary
│   ├── preprocessing.py            # bandpass + ICA + segmentation
│   ├── features.py                 # 207 ميزة (5 time + 4 spectral × 23 ch)
│   ├── cache.py                    # حفظ الميزات لكل مريض
│   ├── models.py                   # RF, SVM, XGB, LSTM, CNN
│   ├── metrics.py                  # accuracy, sens, spec, AUC + Wilson CI
│   ├── loocv.py                    # حلقة LOOCV الرئيسية
│   ├── statistical_tests.py        # NEW: Wilcoxon + bootstrap CI
│   ├── distribution_shift.py       # NEW: MMD + Wasserstein + Mahalanobis
│   └── visualize.py                # رسوم وجداول
└── scripts/
    ├── run_loocv.py                # المسار الرئيسي
    ├── run_statistical_tests.py    # NEW: stats فقط
    └── run_distribution_shift.py   # NEW: shift فقط
```

---

## 8. الكاش متوافق مع v1

ملفات `.npz` التي بنيتَها مع v1 (cache.py لم يتغيّر) تعمل مباشرةً مع v2.
**لا تحتاج إلى إعادة بناء الكاش**. شغّل:

```bash
python scripts/run_loocv.py --skip-cache
```

---

## 9. ما الذي يجب تحديثه في النص بعد تشغيل v2

بعد تشغيل LOOCV الجديد والحصول على نتائج XGBoost والاختبارات الإحصائية،
ستحتاج تحديث الفصول التالية في الأطروحة:

- **Table 4.1**: إضافة عمود XGBoost.
- **§4.1**: إضافة فقرة عن XGBoost ومقارنته بـ RF/SVM (Wilcoxon).
- **§4.1**: إضافة bootstrap CIs بدلاً من mean ± std فقط.
- **§5.1**: استبدال "paired t-test" بـ "Wilcoxon signed-rank with Holm
  correction"، ذكر القيم.
- **§5.2.3**: استبدال الفرضيات النوعية بنتائج Spearman الكمية من
  `distribution_shift.csv`.
- **Abstract، §4.2.2، §6.1**: توحيد قائمة المرضى المنهارين.
- **§3.1.2**: تصحيح وصف chb21/chb23/chb24.

التفاصيل في `CHANGELOG.md` قسم "What is NOT addressed in v2".
