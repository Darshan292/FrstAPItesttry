import uuid
import os

def save_temp_file(file, content):
    file_path = f"/tmp/{uuid.uuid4()}_{file}"
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path