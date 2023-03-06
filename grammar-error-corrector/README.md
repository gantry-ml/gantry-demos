# Gantry

This is a tutorial for Gantry. See [the documentation](https://docs.gantry.io/docs/quickstart) to follow along.
docs.

## Quickstart

1. Set up environment in demo directory

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

2. Update api key using .env.example

    ```bash
    cp .env.example .env
    ```
    
    Now update the `.env` file with your API key.

3. Backfill historical data
    
    ```bash
    python backfill.py --load-data --create-views
    ```


4. Run local Gradio app
    
    ```bash
    python app.py
    ```

5. Open the notebook for data curation
    
    ```bash
    jupyter lab
    ```