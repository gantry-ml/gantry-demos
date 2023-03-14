# CaptiveCaption Crash Course

CaptiveCaption is a silly demo idea meant to be simple enough to emphasize some key concepts in building a Gantry Demo, but complex enough to enable you to build a more complete product demo from scratch.

Specifically, it is meant to demonstrate some key patterns and practices around:

1. Exploring datasets
2. Simulating user behavior
3. Generating predictions with a pretrained ML model
4. Simulating a pattern of user feedback
5. Logging complex data to Gantry
6. Making an interactive application that logs data to Gantry

## Instructions

1. Clone this repo
2. Navigate to the `captive-caption` directory
3. Run `make create_environment` to create a virtual environment
4. Run `make_install_requirements` to install the required packages
4. Run `make configure_jupyter` to configure Jupyter
5. Use the `.env.sample` file to create a `.env` file with your Gantry API key
6. Activate your enironment and run `jupyter lab` to start Jupyter Lab
7. Open the `0.0-captive-caption.ipynb` notebook and enjoy!


## Notes

- The notebook is initially set to run in with `DEV_SAMPLE = True`, in order to speed up the exectuion of the notebook.  This will only use a small sample of the data.  If you want to run the notebook with the full dataset, set `DEV_SAMPLE = False` in the notebook.
- You should consider changing the variable `MY_DEMO_PREFIX` to something unique to you.  This will help you avoid collisions with other users of the Gantry s3 bucket.
- Finally, you should change the `application` variable to something unique to you. This will help you avoid collisions with other Gantry applications.
