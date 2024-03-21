import os
from pprint import pprint

import yaml

with open("./params.yaml") as f:
    params = yaml.safe_load(f)


params_root_folder = "params/"

for model_class in filter(
    lambda k: k.endswith("distributions") and "malnut" not in k, params.keys()
):
    base_name = model_class.replace("_distributions", "")
    for distribution_name in params[model_class].keys():
        for dataset_name in params[model_class][distribution_name].keys():
            model_kwargs = params[model_class][distribution_name][dataset_name].copy()
            model_kwargs.update(distribution=distribution_name)
            fit_kwargs = model_kwargs.pop("fit_kwargs", {})
            model_params = dict(
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
                compile_kwargs={"jit_compile": True},
            )
            print(model_class, dataset_name)
            pprint(model_params)
            if "benchmark" in model_class:
                dataset_type = "benchmark"
            else:
                dataset_type = "sim"
            params_file_name = os.path.join(
                params_root_folder,
                dataset_type,
                dataset_name,
                "_".join((base_name, distribution_name)) + ".yaml",
            )
            os.makedirs(os.path.dirname(params_file_name), exist_ok=True)
            with open(params_file_name, "w+") as f:
                yaml.dump(model_params, f)
