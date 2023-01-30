import gantry
from config import GantryConfig, ModelConfig
from model import ModelWrapper
import gradio as gr
from difflib import Differ
import uuid

gantry.init(api_key=GantryConfig.GANTRY_API_KEY)
model_wrapper = ModelWrapper(ModelConfig.MODEL_DIR)

def correct_grammar(text, join_key):
    corrected_sentence = model_wrapper.infer(text)
    gantry.log_record(
        application = GantryConfig.GANTRY_APP_NAME,
        inputs = {"text": text},
        join_key = join_key,
        outputs = {"inference": corrected_sentence},
        tags = {
            "env": GantryConfig.GANTRY_PROD_ENV,
            "test-tag": "my-test-data"
        }
    )
    return corrected_sentence

def diff_texts(input_text, join_key):
    corrected = correct_grammar(input_text, join_key)
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(input_text, corrected)
    ]

def send_feedback(join_key):
    gantry.log_record(
        application = GantryConfig.GANTRY_APP_NAME,
        feedback = {"correction_accepted": True},
        join_key = join_key,
        tags = {
            "env": GantryConfig.GANTRY_PROD_ENV,
            "test-tag": "my-test-data"
        }
    )

with gr.Blocks() as gec_app:
    join_key = str(uuid.uuid1())
    gr.Markdown("Welcome to the Gantry Grammar Error Corrector!")

    text_input = gr.Textbox(
        label="Text to correct",
        lines=3,
    )
    join_key_input = gr.Textbox(
        label="Feedback",
        value=join_key,
        visible=False,
    )
    text_output = gr.HighlightedText(
        label="Suggestions",
        combine_adjacent=True,
    ).style(color_map={"+": "green", "-": "red"})
    text_button = gr.Button("Submit")
    
    text_button.click(diff_texts, inputs=[text_input, join_key_input], outputs=text_output)

    with gr.Accordion("Send Feedback"):
        gr.Markdown("If you liked the suggestion, tell us about it by clicking accept!")
        feedback_button = gr.Button("Accept suggestions")

    feedback_button.click(send_feedback, inputs=[join_key_input])

if __name__ == "__main__":
    gec_app.launch()