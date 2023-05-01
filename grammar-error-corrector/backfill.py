import argparse
import pandas as pd
import datetime

from config import logger, GantryConfig, DataStorageConfig
import gantry
from gantry.query.time_window import TimeWindow

gantry.init(api_key=GantryConfig.GANTRY_API_KEY, send_in_background=False)
# application = gantry.create_application(GantryConfig.GANTRY_APP_NAME)
application = gantry.get_application(GantryConfig.GANTRY_APP_NAME)

def retrieve_data() -> pd.DataFrame:
    logger.info("Retrieving demo data from public S3 bucket")
    return pd.read_csv(
        DataStorageConfig.DATASET_URL,
        engine="c",
        lineterminator="\n",
    )


def load_to_gantry(df: pd.DataFrame, gantry_env: str):
    logger.info(f"Logging {len(df)} to Gantry environment {gantry_env}")

    # There are two date formats in the dataset; parse them separately
    datetime = pd.to_datetime(
        df["timestamp"],
        format = "%Y-%m-%d %H:%M:%S.%f%z",
        errors="coerce",
        )
    datetime = datetime.fillna(
        pd.to_datetime(
            df["timestamp"],
            format="%Y-%m-%d %H:%M:%S%z",
            errors="coerce"),
        )

    application.log(
        timestamps=[ts.to_pydatetime() for ts in datetime],
        inputs=df[["text", "account_age_days"]],
        row_tags=df[["username"]].to_dict("records"),
        outputs=df["inference"],
        feedbacks=df["correction_accepted"],
        join_keys=df["uuid"].to_list(),
        global_tags={"env": gantry_env},
        as_batch=True,
    )


def _utc_time_helper(year: int, month: int, day: int) -> datetime.datetime:
    return datetime.datetime(year, month, day, 0, 0).astimezone(
        datetime.timezone.utc,
    )


def _correction_filter_helper(correction_accepted: bool) -> list:
    return [
        {
            "boolean_query": correction_accepted,
            "feature_name": "feedback.correction_accepted",
        }
    ]


def _env_filter_helper() -> list:
    return [
        {
            "category_query": [GantryConfig.GANTRY_PROD_ENV],
            "dtype": "tag",
            "feature_name": "env",
        }
    ]


LAST_WEEK_START = _utc_time_helper(2022, 4, 16)
LAST_WEEK_END = _utc_time_helper(
    2022,
    4,
    22,
)
THIS_WEEK_START = LAST_WEEK_END
THIS_WEEK_END = _utc_time_helper(2022, 4, 26)
THIS_WEEK_QUERY, LAST_WEEK_QUERY = "this-week", "last-week"
ACCEPTED_QUERY, REJECTED_QUERY = "accepted", "rejected"


def create_queries():
    # This week query
    logger.info(f"Creating query named {THIS_WEEK_QUERY} for {application._name}")
    query = application.query(
        time_window=TimeWindow(start_time=THIS_WEEK_START, end_time=THIS_WEEK_END),
        filters=_env_filter_helper(),
    )
    application.save_query(THIS_WEEK_QUERY, query)

    # Last week query
    logger.info(f"Creating query named {LAST_WEEK_QUERY} for {application._name}")
    query = application.query(
        time_window=TimeWindow(start_time=LAST_WEEK_START, end_time=LAST_WEEK_END),
        filters=_env_filter_helper(),
    )
    application.save_query(LAST_WEEK_QUERY, query)

    # Accepted query
    logger.info(f"Creating query named {ACCEPTED_QUERY} for {application._name}")
    query = application.query(
        time_window=TimeWindow(start_time=THIS_WEEK_START, end_time=THIS_WEEK_END),
        filters=_env_filter_helper() + _correction_filter_helper(True),
    )
    application.save_query(ACCEPTED_QUERY, query)

    # Rejected query
    logger.info(f"Creating query named {REJECTED_QUERY} for {application._name}")
    query = application.query(
        time_window=TimeWindow(start_time=THIS_WEEK_START, end_time=THIS_WEEK_END),
        filters=_env_filter_helper() + _correction_filter_helper(True),
    )
    application.save_query(REJECTED_QUERY, query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-data", action="store_true")
    parser.add_argument("--create-queries", action="store_true")
    args = parser.parse_args()
    if not (args.load_data or args.create_queries):
        logger.error("Must pass either '--load-data' or '--create-queries'")
    else:
        if args.load_data:
            data = retrieve_data()
            load_to_gantry(data, GantryConfig.GANTRY_PROD_ENV)
        if args.create_queries:
            create_queries()
