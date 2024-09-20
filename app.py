import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Dictionary for emotion emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    # Set the gradient background color from top to bottom
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(to bottom, #1E90FF, #87CEEB);
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput textarea {
            background-color: #000000;
            color: #000;
        }
        .stAlert div {
            background-color: #ffe0b2;
            color: #000;
        }
        .boxed-title {
            border: 2px solid #000;
            padding: 10px;
            border-radius: 10px;
            background-color: #FF6666;
            text-align: left;
            color: #1E90FF;
            margin-bottom: 20px;
        }
        .boxed-subheader {
            border: 2px solid #000;
            padding: 10px;
            border-radius: 10px;
            background-color: #FFC0CB;
            text-align: left;
            color: #4682B4;
        }
        .confidence-box {
            border: 2px solid #000;
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(255, 192, 203, 0.3);
            margin-top: 10px;
        }
        .result-box {
            border: 2px solid #000;
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.5);
            margin-top: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="boxed-title"><h1>Text Emotion Detection</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="boxed-subheader"><h3>Detect Emotions In Text</h3></div>', unsafe_allow_html=True)

    # Number of input boxes
    num_boxes = st.slider("Select the number of input boxes", min_value=1, max_value=5, value=1)

    input_texts = []
    with st.form(key='my_form'):
        for i in range(num_boxes):
            input_text = st.text_area(f"Input Text {i+1}", key=f"text_area_{i}")
            input_texts.append(input_text)
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        for i, raw_text in enumerate(input_texts):
            if raw_text:
                st.write(f"### Result for Input {i+1}")

                col1, col2 = st.columns(2)

                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)

                with col1:
                    st.success("Original Text")
                    st.markdown(
                        f"""
                        <div class="result-box">
                            {raw_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.markdown(
                        f"""
                        <div class="result-box">
                            {prediction}: {emoji_icon}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Adding the confidence score in a styled box
                    st.markdown(
                        f"""
                        <div class="confidence-box">
                            Confidence: {np.max(probability):.2f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.success("Prediction Probability")
                    # Graph container with border around the entire graph area


                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]

                    # Create the graph with a direct black border around it
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(
                        width=300,
                        height=200
                    ).configure_view(
                        stroke=None  # No additional border needed inside the container
                    )

                    st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()

