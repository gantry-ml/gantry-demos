import streamlit as st
import gantry
from diff_match_patch import diff_match_patch as dmp_module
from gec_diff import my_component
from config import GantryConfig, ModelConfig
from model import ModelWrapper

gantry.init(api_key=GantryConfig.GANTRY_API_KEY)
model_wrapper = ModelWrapper(ModelConfig.MODEL_DIR)

def correct_grammar(text):
    corrected_sentence = model_wrapper.infer(text)
    gantry.log_record(
        application=GantryConfig.GANTRY_APP_NAME,
        inputs={"text": text},
        outputs={"inference": corrected_sentence},
        tags={
            "env": GantryConfig.GANTRY_PROD_ENV,
            "test-tag": "my-test-data"
        }
    )
    return corrected_sentence

def build_diff_layout(txt1, txt2):
    dmp = dmp_module()
    diff = dmp.diff_main(txt1, txt2)
    dmp.diff_cleanupSemantic(diff)
    my_component(diff=diff)

st.set_page_config(
     page_title="Grammar by Gantry",
     page_icon="https://app.gantry.io/gantry-logo.ico",
     layout="wide"
 )

html_string = """
    <div style='display:inline-flex;flex-direction:column'>
        <h1 style='padding:0;line-height:2rem'>Grammmar</h1>
        <h6 style='align-self:flex-end;color:rgba(240,74,0,1)'>
            by Gantry
        </h6>
    </div>
"""

st.markdown(html_string, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    code_text = '''
        
import streamlit as st
import gantry
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GANTRY_API_KEY = "GRAMMAR-API-KEY"
GANTRY_ENV = "production"

# We initialize the Gantry client with an API key from our Gantry account
gantry.init(
    api_key=GANTRY_API_KEY
)

# Load our grammatical error correction model (and tokenizer) from Hugging Face 
# https://huggingface.co/prithivida/grammar_error_correcter_v1
@st.cache(allow_output_mutation=True)
def load_gramformer_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return (model, tokenizer)

# This is our function for inference, it's inputs are user's text, it's output is 
# the generated, corrected text from our model.
# 
# The "@gantry.instrument" annotation automatically logs the inputs 
# and return value (the output) of this function to Gantry. 
# No further setup required!
@gantry.instrument('grammmar-app')
def infer(txt):
    # This model requires a task specific prefix ("gec:")
    tokenized_sentence = gramformer_tokenizer("gec: " + txt, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    corrected_sentence = gramformer_tokenizer.decode(
        gramformer_model.generate(
            input_ids = tokenized_sentence.input_ids,
            do_sample=True,
            max_legnth=512,
            top_k=50, 
            top_p=0.95, 
            early_stopping=True,
        )[0],
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

    return corrected_sentence

### STREAMLIT APP ###
 with st.form(key="gec_input"):
    doc = st.text_area(
        "Start writing below",
        height=260
    )

    MAX_WORDS = 128
    import re
    res = len(re.findall(r"\w+", doc))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + f" Only the first {MAX_WORDS} words will be reviewed."
        )

        doc = doc[:MAX_WORDS]

    submit_button = st.form_submit_button(label="✨ Check your text")

    if not submit_button:
        st.stop()
    
    corrected = infer(doc)
    build_diff_layout(doc, corrected)

    '''
    st.code(code_text)

with col2:

    with st.form(key="gec_input"):

        doc = st.text_area(
            "Start writing below",
            height=260
        )

        MAX_WORDS = 512
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + f" Only the first {MAX_WORDS} words will be reviewed."
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Check your text")

    if not submit_button:
        st.stop()

    corrected = correct_grammar(doc)
    build_diff_layout(doc, corrected)
