WhatsApp Spam Detector

A polished machine-learning application for detecting SMS & WhatsApp spam using a scikit-learn Pipeline (TF-IDF + MultinomialNB).
This project goes beyond a basic classifier and includes a presentation-ready UI, animated interactions, batch CSV predictions, prediction history, confetti effects, and a model upload feature — ideal for academic demonstrations, viva, portfolios, and machine-learning showcases.

Developed by
Abhishek Basu, Ananya Raj, Sneha Das, Payal Guin, Subhojit Khamrai

🚀 Features (Expanded)

🔮 Real-time ML Prediction

Enter any message and receive a Spam / Ham prediction instantly.

Processing includes:

Lowercasing

URL removal

Noise removal

Space normalization

TF-IDF vectorization

📊 Confidence Score

If the model supports predict_proba, the app displays:

A probability bar

Exact confidence percentage
Great for explaining ML decision-making in viva/demo.

⚡ Quick-Sample Buttons

One-click message loading for:

Spam examples

Ham examples

Additional custom samples
Useful for fast, smooth demos.

🗂 Prediction History

Stores last 10 messages during the session with:

Text

Prediction

Confidence

Timestamp

📑 Batch CSV Prediction

Upload a .csv file containing a column named message.
The app processes all the rows and returns:

🏗 Architecture

┌──────────────────────────┐
│        User Input         │
│  (SMS / WhatsApp message) │
└───────────────┬──────────┘
                ▼
       ┌────────────────┐
       │ Preprocessing  │
       │ clean_text()   │
       │ - lowercase    │
       │ - remove URLs  │
       │ - remove noise │
       └───────┬────────┘
               ▼
     ┌────────────────────┐
     │ ML Pipeline        │
     │  TF-IDF Vectorizer │
     │  Multinomial NB    │
     └─────────┬──────────┘
               ▼
      ┌──────────────────┐
      │ Prediction Output │
      │ Spam / Ham        │
      │ Confidence Score  │
      └─────────┬────────┘
                ▼
   ┌────────────────────────┐
   │ Streamlit UI Rendering │
   │ Results, animations,   │
   │ history, batch tools   │
   └────────────────────────┘

//Snapshots
<img width="1652" height="807" alt="image" src="https://github.com/user-attachments/assets/aa686408-c841-4e3c-a774-3e8df9950fdd" />

confidence

downloadable result file
