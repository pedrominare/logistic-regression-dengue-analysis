import json
from jsonschema import validate, ValidationError


def json_validator(json_schema, json_arquivo):
    with open(json_schema) as schema_file:
        schema = json.load(schema_file)

    # Carrega o JSON de entrada do arquivo input.json
    with open(json_arquivo) as json_file:
        json_data = json.load(json_file)

    # Valida o JSON
    try:
        validate(instance=json_data, schema=schema)
        print("JSON válido!")
    except ValidationError as e:
        print("JSON inválido:", e.message)
