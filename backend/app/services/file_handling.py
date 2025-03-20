import os
import shutil
import uuid

def save_uploaded_file(file, upload_dir):
    """Save the uploaded file in a session-specific directory"""
    session_id = str(uuid.uuid4())
    user_dir = os.path.join(upload_dir, session_id)
    os.makedirs(user_dir, exist_ok=True)

    file_path = os.path.join(user_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return session_id, file_path
