#!/usr/bin/env python
# coding: utf-8

# # A crash course in useful patterns and practices for building a Gantry demo
# 
# This notebook uses a silly example to provide you with some basic tools useful in constructing a Gantry product demo. Below, we'll cover
# 1. Exploring datasets
# 2. Simulating user behavior
# 3. Generating predictions with a pretrained ML model
# 4. Simulating a pattern of user feedback
# 5. Logging image data to Gantry
# 6. Creating an interactive app using the model and Gantry logging
# 
# With a basic handle on the above, you will be well set-up to explore some of your own ideas, test things out with teamates, and iterate your way to a great demo.
# 
# For now, we're keeping this tutorial simple, restricted only to the above: this notebook does not constitute a valid, complete Gantry demo. Currently, it's more of a technical "how to," but we will frame things around a demo idea.
# 
# ## CaptiveCaption
# 
# *Recommend the best caption for an image, given a list of candidates*
# 
# CaptiveCaption is an ML-powered service built for and used by a (fake) website where users upload picutres and add generally unimaginative captions to them. CaptiveCaption ranks image captions using similarity score relevance and recommends the best caption, typically the most literal or informative one. The top-ranking caption is then displayed on the website, but users can up/downvote the suggested caption.
# 
# The site has become popular with boring animal lovers who write dry captions, but recently there's been an increase in downvotes for the captions suggested by CaptiveCaption. Investigation will show that this is due to many more cat photos coming through lately. We'll find that the model tends to prefer a more literal description for the cat photos than users, who prefer humorous captions for cats. Ultimately, we'll send these photos off for expert labeling and improve the model's sense of humor.
# 
# * INPUTS: Simulate user-provided caption candidates with `"ydshieh/vit-gpt2-coco-en"`, an image-to-text model.
# * OUTPUTS: Rank candidates image-text pairs using `"nvidia/groupvit-gcc-redcaps"` and recommend a best caption
# * FEEDBACK: When it's a cat photo, and the true label isn't the top rank, we'll add "downvoted" feedback.
# 
# * Log to Gantry
#   * images
#   * recommended best caption
#   * best captions relevance score
#   * timestams witrh recent increase in cats
#   * more downvote feedback for cat photos

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import os
from pathlib import Path
import requests
import urllib
import uuid
import warnings

from cloudpathlib import CloudPath
import datasets
from dotenv import dotenv_values
from faker import Faker
import gantry
import gradio as gr
from IPython import display
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoProcessor, GroupViTModel, pipeline, set_seed
from tqdm import tqdm

# Ignore some annoying transformer warnings
warnings.filterwarnings("ignore")


config = dotenv_values("../.env")

# For sending data to s3
os.environ["AWS_PROFILE"] = config.get("AWS_PROFILE", "sandbox")

# For logging to Gantry
GANTRY_API_KEY = config.get("GANTRY_API_KEY_STAGING")
if not GANTRY_API_KEY:
    raise ValueError("Set your Gantry API key!")

# For logging to non-prod Gantry
if config.get("GANTRY_LOGS_LOCATION"):
    os.environ["GANTRY_LOGS_LOCATION"] = config.get("GANTRY_LOGS_LOCATION")
    logger.info(f"Set Gantry logging location: {os.environ['GANTRY_LOGS_LOCATION']}")

# Wrap column text in Pandas
pd.set_option("display.max_colwidth", 0)


# # Explore the data

# First, let's stream a subset of data that we can use for our demo. We'll be using the [Red Caps](https://huggingface.co/datasets/red_caps) dataseet from Hugging Face because it has real image-caption pairs from Reddit that we can use as a basis for simulating users, generating predictions, feedback, and more. Plus, Hugging Face hosts some  odels trained on this dataset, as we'll see below.
# 
# > RedCaps is a large-scale dataset of 12M image-text pairs collected from Reddit. Images and captions from Reddit depict and describe a wide variety of objects and scenes. The data is collected from a manually curated set of subreddits (350 total), which give coarse image labels and allow steering of the dataset composition without labeling individual instances. RedCaps data is created by the people, for the people – it contains everyday things that users like to share on social media, for example hobbies (r/crafts) and pets (r/shiba). Captions often contain specific and fine-grained descriptions (northern cardinal, taj mahal). Subreddit names provide relevant image labels (r/shiba) even when captions may not (mlem!), and sometimes may group many visually unrelated images through a common semantic meaning (r/perfectfit).
# 
# **12 Million** is a lot of pairs, so let's stream the dataset.

HF_DATASET_NAME = "red_caps"
CAT_SUBSET = "cats"  # subsets listed on dataset page
ALL_ANIMALS_SUBSET = "alltheanimals"


cats_streamed = datasets.load_dataset(
    HF_DATASET_NAME,
    CAT_SUBSET,
    split="train",  # we can choose any valid split
    streaming=True,
)

animals_streamed = datasets.load_dataset(
    HF_DATASET_NAME,
    ALL_ANIMALS_SUBSET,
    split="train",  # we can choose any valid split
    streaming=True,
)


# Make the data more user-friendly by `.take`ing some into a Pandas DataFrame.

# **NOTE: The `DEV_SAMPLE` constant can be used for notebook development.**

######## SET T0 FALSE FOR LARGER SAMPLE ########
DEV_SAMPLE = True
################################################

NUM_CAT_SAMPLES = 200
NUM_ANIMALS_SAMPLES = 300

cat_df = pd.DataFrame([record for record in cats_streamed.take(NUM_CAT_SAMPLES)])
cat_df["is_cat"] = True

animal_df = pd.DataFrame(
    [record for record in animals_streamed.take(NUM_ANIMALS_SAMPLES)]
)
animal_df["is_cat"] = False

df = pd.concat([cat_df, animal_df], axis=0).sample(frac=1, random_state=0)

if DEV_SAMPLE:
    df = df.sample(50)

df.head()


# Some urls might not exist exists. Let's get rid of the ones that don't.

def image_exists(row, new_col="image_exists"):
    try:
        urllib.request.urlopen(row.image_url)
        row[new_col] = True
    except Exception as e:
        row[new_col] = False
    return row


tqdm.pandas(desc="Check for urls")
df = df.progress_apply(image_exists, axis=1)
num_failed = (~df.image_exists).sum()
logger.info(f"{num_failed} of {len(df)} image urls failed. Dropping {num_failed}...")
df = df[df.image_exists].reset_index(drop=True)  # avoid weird index errors


# ## Look at images!

def show_image(url, width=300):
    image = Image.open(requests.get(url, stream=True).raw)
    if width:
        ratio = width / image.width
        height = int(image.height * ratio)
        image = image.resize((width, height), Image.LANCZOS)
    return image


def show_image_caption_pair(idx=0):
    print(df.caption[idx])
    return show_image(df.image_url[idx])


show_image_caption_pair(np.random.choice(len(df)))


show_image_caption_pair(np.random.choice(len(df)))


show_image_caption_pair(np.random.choice(len(df)))


show_image_caption_pair(np.random.choice(len(df)))


# # Simulate user captions
# 
# This is all well and good, but remember, our model ranks captions provided by users to suggest the best caption. This means **we need "user-submitted" captions to rank**. Since each image in the dataset only comes with one caption, we'll need to generate the others.
# 
# Once way we can do that is to use an ML model to generate alternatives for us!
# 
# Hugging Face [pipelines](https://huggingface.co/docs/transformers/pipeline_tutorial) make it easy to get going. Consider the following image.

num_captions_to_generate = 4
captioner = pipeline(
    "image-to-text",
    model="ydshieh/vit-gpt2-coco-en",
    model_kwargs={
        "num_return_sequences": num_captions_to_generate,
        "num_beams": num_captions_to_generate,
    },
)


def generate_captions(url, captioner=captioner):
    return [caption["generated_text"] for caption in captioner(url)]


idx = 6
show_image_caption_pair(idx)


rec_captions = generate_captions(df.image_url[idx])
rec_captions


# # Use model to rank user captions
# 
# [This model](https://huggingface.co/nvidia/groupvit-gcc-redcaps) computes similarity scores between text-image pairs. The text that is "the most similar" to the image is considered "the best caption."

PROCESSOR = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-redcaps")
MODEL = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-redcaps")


def rank_captions(
    captions, image_url=None, image=None, processor=PROCESSOR, model=MODEL
):
    if image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(
        text=captions,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze().detach().numpy().tolist()
    results = {caption: prob for caption, prob in zip(captions, probs)}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


# Let's see how our captions are ranked.

ranked_captions = rank_captions(rec_captions, df.image_url[idx])
ranked_captions


# Add in the true caption

ranked_captions = rank_captions(
    rec_captions + [df.caption[idx]],
    df.image_url[idx],
)
ranked_captions


# In this example, the true caption is the lowest rank. It's this kind of pattern we want to exploit for negative feedback signals.
# 
# First, let's add our predictions to the dataset.

def get_best_caption(ranked_captions):
    best_caption_text = max(ranked_captions, key=ranked_captions.get)
    best_caption_prob = max(ranked_captions.values())
    return best_caption_text, best_caption_prob


get_best_caption(ranked_captions)


def add_preds_to_dataset(row):
    rec_captions = generate_captions(row.image_url)
    if row.is_cat:
        rec_captions.append(row.caption)
    ranked_captions = rank_captions(rec_captions, row.image_url)
    best_caption_text, best_caption_prob = get_best_caption(ranked_captions)
    row["best_caption_text"] = best_caption_text
    row["best_caption_prob"] = best_caption_prob
    return row


tqdm.pandas(desc="Add preds to dataset")
df = df.progress_apply(add_preds_to_dataset, axis=1)


# # Create feedback
# 
# For cat photos, when the best caption isn't the true caption, we'll downvote the caption. This simulates behavior in which users are dissatisfied with the caption.

def feedback_downvote_if_not_caption_for_cats(
    row, feedback_col="downvoted", join_key="uuid"
):
    if row.is_cat and (row.best_caption_text != row.caption):
        row[feedback_col] = True
    else:
        row[feedback_col] = False

    row[join_key] = str(uuid.uuid4())
    return row


df = df.apply(feedback_downvote_if_not_caption_for_cats, axis=1)
df.downvoted.mean()


# # Log to Gantry

# ## Send images to s3
# 
# Since this data includes images, we'll need to first send the images to s3. This is because currently, Gantry is configured to allow customers to log images by providing Gantry with s3 URIs via the logger and some limited read access policies that must be configured on the customer end.
# 
# Gantry employees can send images to the `gantry-customer-bucket-access-test-pdx` bucket using their `sandbox` profile. URIs from this bucket can be rendered in the Gantry front end.
# 
# Images can be moved to s3 in any number of ways, but we're going to use [cloudpathlib](https://cloudpathlib.drivendata.org/stable/) because the only thing better than `pathlib` is **`pathlib` in the cloud**.

####### CHANGE FOR YOUR USE CASE #######

MY_DEMO_PREFIX = "casey-captive-caption"

########################################


BUCKET_NAME = "gantry-customer-bucket-access-test-pdx"
BUCKET_PATH = CloudPath(f"s3://{BUCKET_NAME}")
assert BUCKET_PATH.exists()

DEMO_PATH = BUCKET_PATH / MY_DEMO_PREFIX
DEMO_PATH.mkdir(exist_ok=True)


# Add s3 URIs to DataFrame
IMAGE_URI_COL = "image_s3_uri"
df[IMAGE_URI_COL] = df.image_url.map(lambda url: str(DEMO_PATH / Path(url).name))


for _, row in tqdm(df.iterrows(), total=len(df)):
    with tempfile.NamedTemporaryFile() as temp:
        temp_path = Path(temp.name)
        urllib.request.urlretrieve(row.image_url, temp.name)
        with temp_path.open("rb") as image_file:
            image_bytes = image_file.read()
            with CloudPath(row[IMAGE_URI_COL]).open("wb") as cloud_image_file:
                cloud_image_file.write(image_bytes)


# Now our data is in s3, and our Data Frame has the s3 uri we need to log to Gantry!

df[IMAGE_URI_COL].tolist()[:5]


# ## Controlling the data: add some timestamps and metadata
# 
# Often, we'll want to control *when*  or *where* our demo data "occurred." This allows us to say things like "Oh, we see that on Tuesday, Califronia users started rejecting our predicitons at a much higher rate. Something must be going on!"
# 
# The easiest way to add data like this, and lots of other cool fake stuff, is using the `faker` [library](https://faker.readthedocs.io/en/master/#basic-usage).

Faker.seed(9001)
faker = Faker()

# cat photos just started appearing yesterday
df["timestamp"] = None
df.loc[~df.is_cat, "timestamp"] = sorted(
    [
        faker.date_time_between(
            start_date="-3d",
            tzinfo=datetime.timezone.utc,
        )
        for _ in range((~df.is_cat).sum())
    ]
)
df.loc[df.is_cat, "timestamp"] = sorted(
    [
        faker.date_time_between(
            start_date="-1d",
            tzinfo=datetime.timezone.utc,
        )
        for _ in range((df.is_cat).sum())
    ]
)

df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)

# random states for everyone
df["state"] = [faker.state() for _ in range(len(df))]

df[["timestamp", "state", "caption"]].head()


# ## Send bacthes to Gantry
# Let's log to Gantry!
# 
# First, determine what columns to log

TIMESTAMP_COL = ["timestamp"]

INPUTS = ["uuid", IMAGE_URI_COL, "state"]

OUTPUTS = [
    "best_caption_text",
    "best_caption_prob",
]

FEEDBACK = ["downvoted"]

COLS_TO_KEEP = TIMESTAMP_COL + INPUTS + OUTPUTS + FEEDBACK

to_gantry = df[COLS_TO_KEEP]
to_gantry.head()


gantry.init(GANTRY_API_KEY)

DEMO_APPLICATION = "casey-captive-caption"
gantry.log_records(
    application=DEMO_APPLICATION,
    inputs=to_gantry[INPUTS],
    outputs=to_gantry[OUTPUTS],
    feedbacks=to_gantry[FEEDBACK],
    timestamps=to_gantry.timestamp.dt.to_pydatetime().tolist(),
    tags={"env": "dev-sample"},
    as_batch=True,
)


# # Putting it all together: Make an interactive demo!
# 
# [Gradio](https://gradio.app/quickstart/) makes it super easy for us to demo CaptiveCaption with live Gantry logging. The few lines of code below will create an in-notebook app where
# * users can upload images directly
# * our simulated captions will be generated automatically
# * the resulting captions will be ranked
# * the image will be uploaded to s3
# * all of the relvant data will be logged to Gantry under the `gradio` environment tag

def upload_image_to_s3(image, s3_prefix=DEMO_PATH):
    with tempfile.NamedTemporaryFile(suffix=".JPEG") as temp:
        temp_path = Path(temp.name)
        image.save(temp)
        with temp_path.open("rb") as image_file:
            image_bytes = image_file.read()
            image_uri = str(s3_prefix / temp_path.name)
            with CloudPath(image_uri).open("wb") as cloud_image_file:
                cloud_image_file.write(image_bytes)
    return image_uri


def predict(image, user_caption=None, image_uri_col=IMAGE_URI_COL):
    rec_captions = generate_captions(image)
    if user_caption:
        rec_captions.append(user_caption)
    # note we're passing a PIL image directly instead of an url
    ranked_captions = rank_captions(rec_captions, image=image)
    best_caption_text, best_caption_prob = get_best_caption(ranked_captions)

    # Log to Gantry
    image_uri = upload_image_to_s3(image)
    gantry.log_record(
        application=DEMO_APPLICATION,
        inputs={
            "uuid": str(uuid.uuid4()),
            "state": faker.state(),
            image_uri_col: image_uri,
        },
        outputs={
            "best_caption_text": best_caption_text,
            "best_caption_prob": best_caption_prob,
        },
        # no feedback hook right now!
        tags={"env": "gradio"},
    )
    return ranked_captions


demo = gr.Interface(
    title="CaptiveCaption",
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Text(placeholder="See how your own caption ranks against generated ones!"),
    ],
    outputs=gr.Label(num_top_classes=5),
)
demo.launch()


# # What's next?
# - cache the processed results for faster demo deployment
# - improve the strngth of the signal for the problem in Gantry
# - **bonus: close the loop** how do we *curate* a dataset of cat photos?



