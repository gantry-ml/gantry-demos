


import os
import datasets
from datasets import Value, Sequence
import pandas as pd
import yaml

_DESCRIPTION = """\
caseygectestnewaccountcurator1411474 huggingface dataset
"""

# TODO: enable is we need to specify the license
# _LICENSE = ""


_URLS = {
    "train": ['/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-24T00:00:00_rows_0_999_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-08T00:00:00_rows_0_329_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-01T00:00:00_rows_0_51_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-30T00:00:00_rows_0_67_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-13T00:00:00_rows_0_294_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-05T00:00:00_rows_0_241_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-10T00:00:00_rows_0_405_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-25T00:00:00_rows_0_999_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-16T00:00:00_rows_0_343_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-28T00:00:00_rows_0_379_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-07T00:00:00_rows_0_85_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-19T00:00:00_rows_0_529_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-18T00:00:00_rows_0_139_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-17T00:00:00_rows_0_162_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-23T00:00:00_rows_0_999_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-02T00:00:00_rows_0_88_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-20T00:00:00_rows_0_871_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-15T00:00:00_rows_0_492_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-11T00:00:00_rows_0_310_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-04T00:00:00_rows_0_141_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-26T00:00:00_rows_0_310_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-29T00:00:00_rows_0_160_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-12T00:00:00_rows_0_333_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-22T00:00:00_rows_0_477_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-06T00:00:00_rows_0_184_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-14T00:00:00_rows_0_333_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-09T00:00:00_rows_0_521_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-21T00:00:00_rows_0_301_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-03T00:00:00_rows_0_124_selector_0.csv', '/Users/casey/repos/gantry/gantry-demos/grammar-error-corrector/casey-gec-test-new-account-curator-1411474/tabular_manifests/raw_data_2022-04-27T00:00:00_rows_0_445_selector_0.csv'],
    # "test": ,
    # "eval": ,
}


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class caseygectestnewaccountcurator1411474(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features({'__time': Value(dtype='timestamp[ns, tz=UTC]', id=None), 'inputs.account_age_days': Value(dtype='int64', id=None), 'inputs.text': Value(dtype='string', id=None), 'outputs.inference': Value(dtype='string', id=None), 'feedback.correction_accepted': Value(dtype='bool', id=None)})

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            
            # license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        # TODO: replace the download method with S3 url download so we can download directly from S3
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepaths": downloaded_files["train"]},
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={"filepaths": downloaded_files["test"]},
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={"filepaths": downloaded_files["eval"]},
            # ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepaths):
        # This method handles input defined in _split_generators to yield (key, example)
        # tuples from the dataset. The `key` is for legacy reasons (tfds) and is not important in
        # itself, but must be unique for each example.
        converters = {
            feature_name: yaml.safe_load
            for feature_name, feature_type in {'__time': Value(dtype='timestamp[ns, tz=UTC]', id=None), 'inputs.account_age_days': Value(dtype='int64', id=None), 'inputs.text': Value(dtype='string', id=None), 'outputs.inference': Value(dtype='string', id=None), 'feedback.correction_accepted': Value(dtype='bool', id=None)}.items() if isinstance(feature_type, Sequence)
        }

        cur_idx = -1
        for filepath in filepaths:
            dict_from_csv = pd.read_csv(filepath, index_col=False, converters=converters).to_dict("records")
            for record in dict_from_csv:
                cur_idx += 1
                yield cur_idx, record