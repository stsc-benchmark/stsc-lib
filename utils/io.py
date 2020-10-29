import gzip
import json
import os


def module_path():
    return [entry for entry in os.environ['PYTHONPATH'].split(";") if os.path.split(entry)[1] == "stsc-lib"][0]


def compress_data(data, file_path):
    json_str = json.dumps(data) + "\n"
    json_bytes = json_str.encode("utf-8")

    if not os.path.exists(os.path.split(file_path)[0]):
        os.mkdir(os.path.split(file_path)[0])

    with gzip.GzipFile(file_path, "w") as gz_file:
        gz_file.write(json_bytes)


def decompress_data(file_path):
    #print("loading ", file_path)
    with gzip.GzipFile(file_path, "r") as gz_file:
        json_bytes = gz_file.read()

    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)
