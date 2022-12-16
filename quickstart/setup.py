import argparse
import pandas as pd
import datetime
import boto3
import tempfile

from config import logger, GantryConfig, DataStorageConfig
import gantry
import gantry.query as gantry_query

gantry.init(api_key=GantryConfig.GANTRY_API_KEY, send_in_background=False)

def retrieve_data() -> pd.DataFrame:
    logger.info("Retrieving demo data from public S3 bucket")
    tempfile_handle = tempfile.NamedTemporaryFile()
    s3_client = boto3.client('s3')
    s3_client.download_file(DataStorageConfig.S3_BUCKET, DataStorageConfig.S3_OBJECT, tempfile_handle.name)
    df = pd.read_csv(tempfile_handle.name, engine="c", lineterminator='\n')
    tempfile_handle.close()
    return df

def load_to_gantry(df: pd.DataFrame, gantry_env: str):
    logger.info(f"Logging {len(df)} to Gantry environment {gantry_env}")
    gantry.log_records(
        application=GantryConfig.GANTRY_APP_NAME, 
        timestamps=[ts.to_pydatetime() for ts in pd.to_datetime(df["timestamp"])],
        inputs=df[["text", "account_age_days", "username"]],
        outputs=df["inference"],
        feedbacks=df["correction_accepted"],
        feedback_ids=df["uuid"].to_list(),
        tags={"env": gantry_env},
        as_batch=True
    )

def _utc_time_helper(year: int, month: int, day: int) -> datetime.datetime:
    return datetime.datetime(year, month, day, 0, 0).astimezone(datetime.timezone.utc)

def _correction_filter_helper(correction_accepted: bool) -> list:
    return [{"boolean_query": correction_accepted, "feature_name": "feedback.correction_accepted"}]

def _env_filter_helper() -> list:
    return [{"category_query": [GantryConfig.GANTRY_PROD_ENV], "dtype": "tag", "feature_name": "env"}]

LAST_WEEK_START, LAST_WEEK_END = _utc_time_helper(2022, 4, 16), _utc_time_helper(2022, 4, 22)
THIS_WEEK_START, THIS_WEEK_END = LAST_WEEK_END, _utc_time_helper(2022, 4, 26)
THIS_WEEK_VIEW, LAST_WEEK_VIEW = "this-week", "last-week"
ACCEPTED_VIEW, REJECTED_VIEW = "accepted", "rejected"

def create_views(application_name: str):
    # This week view
    logger.info(f"Creating view named {THIS_WEEK_VIEW} for {application_name}")
    gantry_query.create_view(
        application=application_name,
        name=THIS_WEEK_VIEW,
        start_time=THIS_WEEK_START,
        end_time=THIS_WEEK_END,
        data_filters=_env_filter_helper()
    )
    # Last week view
    logger.info(f"Creating view named {LAST_WEEK_VIEW} for {application_name}")
    gantry_query.create_view(
        application=application_name,
        name=LAST_WEEK_VIEW,
        start_time=LAST_WEEK_START,
        end_time=LAST_WEEK_END,
        data_filters=_env_filter_helper()
    )

    # Accepted view
    logger.info(f"Creating view named {ACCEPTED_VIEW} for {application_name}")
    gantry_query.create_view(
        application=application_name,
        name=ACCEPTED_VIEW,
        start_time=THIS_WEEK_START,
        end_time=THIS_WEEK_END,
        data_filters=_env_filter_helper() + _correction_filter_helper(True) 
    )

    # Rejected view
    logger.info(f"Creating view named {REJECTED_VIEW} for {application_name}")
    gantry_query.create_view(
        application=application_name,
        name=REJECTED_VIEW,
        start_time=THIS_WEEK_START,
        end_time=THIS_WEEK_END,
        data_filters=_env_filter_helper() + _correction_filter_helper(True)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-data", action="store_true")
    parser.add_argument("--create-views", action="store_true")
    args = parser.parse_args()
    if not (args.load_data or args.create_views):
        logger.error("Must pass either '--load-data' or '--create-views'")
    else:
        if args.load_data:
            data = retrieve_data()
            load_to_gantry(data, GantryConfig.GANTRY_PROD_ENV)
        if args.create_views:
            create_views(GantryConfig.GANTRY_APP_NAME)
    
