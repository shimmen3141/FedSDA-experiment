from federated_drift_experiment.parameter_schema import (
    PARAMETERS,
    PARAMETER_SCHEMA_VERSION,
    cli_option,
    parameter,
    paper_symbol,
)
from federated_drift_experiment import config


def test_parameter_schema_has_unique_stable_identifiers():
    assert PARAMETER_SCHEMA_VERSION == 1
    assert len({item.id for item in PARAMETERS}) == len(PARAMETERS)
    exposed_cli_names = [item.cli_name for item in PARAMETERS if item.cli_name]
    assert len(set(exposed_cli_names)) == len(exposed_cli_names)
    assert all(item.csv_name == item.id for item in PARAMETERS)
    assert all(hasattr(config, item.code_name) for item in PARAMETERS)


def test_parameter_schema_maps_code_csv_paper_and_cli_names():
    spec = parameter("aggregation_interval")
    assert spec.code_name == "AGGREGATION_INTERVAL"
    assert spec.csv_name == "aggregation_interval"
    assert paper_symbol(spec.id) == "A"
    assert cli_option(spec.id) == "--aggregation-intervals"


def test_shared_code_value_keeps_method_specific_semantics():
    fedsda = parameter("fedsda_distance_threshold")
    feddrift = parameter("feddrift_distance_threshold")
    assert fedsda.code_name == "FEDSDA_DISTANCE_THRESHOLD"
    assert feddrift.code_name == "FEDDRIFT_DISTANCE_THRESHOLD"
    assert fedsda.csv_name != feddrift.csv_name
    assert fedsda.paper_symbol == "γ"
    assert feddrift.paper_symbol == "δ_FedDrift"
