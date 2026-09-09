# EEG Seizure Detection — LOOCV Pipeline

Pipeline كامل لتنفيذ Leave-One-Patient-Out Cross-Validation على قاعدة CHB-MIT
لمشروع رسالة "EEG Seizure Detection Algorithm Development".

هذا الكود يُصلح القيود المنهجية الرئيسية في النسخة الأصلية من الرسالة:
- يقيّم على **22 مريضاً** بدلاً من مريض واحد.
- ينفّذ **ICA حقيقي** مع رفض المكونات الاصطناعية (وليس مجرد تحلّل بدون رفض).
- يدعم class weighting و early stopping و normalization لـ LSTM/CNN.
- يولّد جداول ورسومات جاهزة لإدراجها في الرسالة مباشرة.

---

## 1. متطلبات النظام

- Python 3.10+
- 16 GB RAM كحد أدنى (32 GB موصى به)
- 50 GB مساحة على القرص (للبيانات + الكاش)
- GPU (اختياري لكن يقلّص زمن LSTM/CNN من ساعات إلى دقائق)

---

## 2. تنزيل البيانات

```bash
# تنزيل CHB-MIT من PhysioNet (~40 GB)
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# الناتج يكون في:
# physionet.org/files/chbmit/1.0.0/chb01/
# physionet.org/files/chbmit/1.0.0/chb02/
# ...
```

---

## 3. التثبيت

```bash
cd seizure_pipeline
python -m venv venv
source venv/bin/activate           # على ويندوز: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. الإعداد

افتح `config.py` وعدّل:

```python
CHB_MIT_PATH = Path("/home/user/physionet.org/files/chbmit/1.0.0")
```

كل المعلمات الأخرى (الحدود الترددية، عدد المكونات في ICA، hyperparameters
الموديلات…) موجودة في نفس الملف ومعلّقة بشكل واضح.

---

## 5. التشغيل

### اختبار سريع (مريض واحد فقط، ~10 دقائق)

```bash
python scripts/run_loocv.py --smoke
```

### تشغيل كامل

```bash
python scripts/run_loocv.py
```

### تشغيل LOOCV فقط (إذا الكاش جاهز من قبل)

```bash
python scripts/run_loocv.py --skip-cache
```

### إعادة بناء الكاش بالقوّة

```bash
python scripts/run_loocv.py --force-cache
```

---

## 6. الزمن المتوقع

| المرحلة | CPU فقط | + GPU |
|---------|---------|-------|
| بناء الكاش (22 مريضاً) | 4–8 ساعات | 4–8 ساعات (لا يستفيد من GPU) |
| LOOCV - RF/SVM فقط | ~1 ساعة | ~1 ساعة |
| LOOCV - مع LSTM/CNN | 8–15 ساعة | 1–3 ساعات |

نصيحة: شغّل بناء الكاش مرة واحدة، ثم استخدم `--skip-cache` لإعادة تجريب
hyperparameters الموديلات بسرعة.

---

## 7. المخرجات

كل المخرجات في `results/`:

```
results/
├── cache/
│   ├── chb01.npz          # ميزات + إشارات خام لكل مريض
│   ├── chb02.npz
│   └── ...
├── tables/
│   ├── loocv_per_fold.csv          # كل fold على حدة (88 صف = 22 مريض × 4 موديلات)
│   ├── loocv_summary.csv           # mean/std/count لكل موديل
│   └── thesis_headline_table.csv   # الجدول الرئيسي للرسالة
├── figures/
│   ├── loocv_boxplots.png          # توزيع الأداء عبر الـ folds
│   ├── per_patient_accuracy.png    # دقة كل مريض على حدة
│   └── aggregated_confusion.png    # confusion matrices مجمّعة
└── logs/
    └── run_loocv.log
```

---

## 8. هيكل الكود

```
seizure_pipeline/
├── config.py                 # كل الإعدادات في مكان واحد
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py        # قراءة EDF + ملفات الـ summary
│   ├── preprocessing.py      # bandpass + ICA حقيقي + segmentation
│   ├── features.py           # 207 ميزة + ICC + PSD
│   ├── models.py             # RF, SVM, LSTM, CNN
│   ├── metrics.py            # accuracy, sens, spec, AUC, Wilson CI
│   ├── cache.py              # حفظ ميزات كل مريض على القرص
│   ├── loocv.py              # حلقة LOOCV الرئيسية
│   └── visualize.py          # رسوم وجداول
└── scripts/
    └── run_loocv.py          # نقطة البدء
```

---

## 9. ما الذي يضمنه هذا الـ pipeline

**صحة منهجية**: كل segment لمريض معيّن إما في التدريب أو الاختبار، أبداً
في الاثنين. لا يوجد تسريب بيانات بين folds.

**توازن الفئات**: مجموعة الاختبار متوازنة دائماً (50/50) لكل مريض، تماماً
كما في الرسالة الأصلية. استراتيجية التدريب قابلة للتغيير من `config.py`.

**استنساخ النتائج**: جميع البذور العشوائية محسوبة من `RANDOM_SEED + fold_idx`
لتكون النتائج قابلة للاستنساخ تماماً.

**تشخيص**: لكل fold، السجل يطبع عدد segments التدريب والاختبار، عدد
ictal، وعدد المكونات التي رفضها ICA.

---

## 10. ماذا بعد تنفيذ LOOCV

بعد الحصول على الجداول والرسومات:
1. أرسلها لي (CSV ملفات + PNG).
2. سأحدّث الرسالة بالقيم الفعلية في الجداول والنصوص ذات العلاقة.
3. سنعيد كتابة فصل النتائج والمناقشة بناءً على الأداء الفعلي عبر 22 مريضاً.

---

## ملاحظات تقنية مهمّة

- بعض المرضى في CHB-MIT لديهم نوبات صرع قليلة جداً (chb22 لديه 3 فقط).
  لو لم يحوي مريض الاختبار أيّ نوبة، يتم تخطّي الـ fold بسلام.
- ملف `chb12-summary.txt` يحتوي بعض الأخطاء التاريخية في PhysioNet؛
  الـ parser يتسامح معها ويُسجّل تحذيراً.
- المريض chb24 هو إعادة تسجيل لـ chb01؛ مُستثنى افتراضياً لتجنّب التداخل.
